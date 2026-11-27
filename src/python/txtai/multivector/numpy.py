"""
NumPy multi-vector module
"""

from ..pipeline import LateEncoder
from ..util import Library

from .base import MultiVector

library = Library()
np, torch, safetensors = library.numpy(), library.torch(), library.safetensors()


class NumPy(MultiVector):
    """
    Stores token rows as per-row symmetric int8 with one fp16 scale per row.
    Int64 offsets mark where each document's token rows start and end.
    """

    # Maximum documents scored in one search batch
    batchsize = 128

    def __init__(self, embeddings, config):
        """
        Creates a store with its own late interaction encoder.

        Args:
            embeddings: owning embeddings instance
            config: late encoder configuration
        """

        super().__init__(embeddings, config)

        # Component settings are not late encoder arguments
        self.encoder = LateEncoder(**{key: value for key, value in config.items() if key not in ("backend", "columns")})

        # Initialize token arrays and document row boundaries
        self.tokens = np.empty((0, 0), dtype=np.int8)
        self.scales = np.empty(0, dtype=np.float16)
        self.offsets = np.zeros(1, dtype=np.int64)
        self.buffers = {}

    def insert(self, documents, index):
        """
        Encodes documents at their internal ANN index ids.

        Args:
            documents: list of (id, data, tags)
            index: internal indexid offset
        """

        # Select documents yielded for vectorization
        batch = []
        for _, document, _ in documents:
            if isinstance(document, dict):
                field = self.text if document.get(self.text) is not None else self.object
                document = document.get(field)
            if document is not None:
                if isinstance(document, list):
                    document = " ".join(document)

                # Non-text objects occupy an empty token slot
                batch.append(document if isinstance(document, str) else None)

        # Encode text and collect quantized rows for this batch
        texts = [text for text in batch if text is not None]
        vectors = iter(())
        if texts:
            vectors = iter(self.encoder.encode(texts, "data").detach().cpu().numpy())
        tokens, scales, offsets = [], [], [int(self.offsets[-1])]
        for text in batch:
            count = 0
            if text is not None:
                rows = next(vectors)
                rows = rows[np.abs(rows).sum(axis=-1) > 0]
                absmax = np.maximum(np.abs(rows).max(axis=1), 1e-12)
                tokens.append(np.clip(np.rint(rows / absmax[:, None] * 127), -127, 127).astype(np.int8))
                scales.append(absmax.astype(np.float16))
                count = len(rows)
            offsets.append(offsets[-1] + count)

        # Append batch rows into buffers and publish populated views
        if tokens:
            self.tokens = self.extendrows("tokens", np.concatenate(tokens))
            self.scales = self.extendrows("scales", np.concatenate(scales))

        # Preserve empty slots for preceding non-text objects
        padding = np.full(index + 1 - len(self.offsets), self.offsets[-1], dtype=np.int64)
        values = np.concatenate((padding, np.asarray(offsets[1:], dtype=np.int64)))
        self.offsets = self.extendrows("offsets", values)

    def extendrows(self, name, values):
        """
        Appends values to an owned buffer and publishes its populated view.

        Args:
            name: array attribute to extend
            values: batch values to append

        Returns:
            populated view of the buffer
        """

        # Match concatenate's shape and dtype
        array = getattr(self, name)
        if name == "tokens" and array.shape[1] == 0:
            array = values[:0]
        if array.shape[1:] != values.shape[1:]:
            raise ValueError("Batch trailing shape differs from stored rows")
        dtype = np.result_type(array.dtype, values.dtype)

        # Reuse only the published view or copy into the next capacity step
        published, buffer = self.buffers.get(name, (None, array))
        size = len(array) + len(values)
        if array is not published or size > len(buffer) or buffer.dtype != dtype:
            buffer = np.empty((max(size, 2 * len(array)), *values.shape[1:]), dtype=dtype)
            buffer[: len(array)] = array
        buffer[len(array) : size] = values
        self.buffers[name] = (buffer[:size], buffer)
        return self.buffers[name][0]

    def delete(self, ids):
        """
        Removes token rows while retaining internal ANN id slots.

        Args:
            ids: internal index ids to delete
        """

        # Mark deleted row ranges and clear their document counts
        counts = np.diff(self.offsets)
        keep = np.ones(len(self.tokens), dtype=bool)
        for index in set(ids):
            start, end = self.offsets[index : index + 2]
            keep[start:end] = False
            counts[index] = 0

        # Compact each array and recompute document boundaries once
        self.tokens = self.tokens[keep]
        self.scales = self.scales[keep]
        self.offsets = self.offsets.copy()
        self.offsets[1:] = np.cumsum(counts)
        self.buffers.clear()

    def score(self, query, ids, limit=None):
        """
        Scores a query against supplied external candidate ids.
        Unknown, deleted and non-text ids are omitted.

        Args:
            query: query text
            ids: external candidate ids
            limit: maximum comparisons to return, defaults to all

        Returns:
            list of (id, score) sorted descending
        """

        # Resolve supplied external ids to nonempty internal slots
        ids = list(ids)
        if not ids:
            return []

        embeddings = self.embeddings
        if embeddings.database:
            candidates = embeddings.database.ids(ids)
        else:
            ids = set(ids)
            candidates = [(index, uid) for index, uid in enumerate(embeddings.ids) if uid is not None and uid in ids]
        candidates = [(index, uid) for index, uid in candidates if self.offsets[index + 1] > self.offsets[index]]
        if not candidates:
            return []

        # Build candidate padding in numpy before one transfer to the query's dtype and device
        queries = self.encoder.encode([query], "query")
        counts = [self.offsets[index + 1] - self.offsets[index] for index, _ in candidates]
        data = np.zeros((len(candidates), max(counts), self.tokens.shape[1]), dtype=np.float32)
        for position, (index, _) in enumerate(candidates):
            start, end = self.offsets[index : index + 2]
            rows = self.tokens[start:end].astype(np.float32)
            scales = self.scales[start:end].astype(np.float32)
            vectors = rows * (scales[:, None] / 127)
            data[position, : end - start] = vectors
        data = torch.from_numpy(data).to(queries)

        # Score stored candidates through the shipped late encoder
        scores = self.encoder.score(queries, data, limit)[0]
        return [(candidates[index][1], score) for index, score in scores]

    def search(self, query, limit=None):
        """
        Computes exact MaxSim over all documents with stored rows.
        Scores fixed-size batches and merges their top results.

        Args:
            query: query text
            limit: maximum results to return, defaults to all

        Returns:
            list of (id, score) sorted descending
        """

        if not self.tokens.size:
            return []

        embeddings = self.embeddings
        queries = self.encoder.encode([query], "query")
        results = []
        for offset in range(0, len(self.offsets) - 1, self.batchsize):
            indices = [
                index for index in range(offset, min(offset + self.batchsize, len(self.offsets) - 1)) if self.offsets[index + 1] > self.offsets[index]
            ]
            if indices:
                counts = [self.offsets[index + 1] - self.offsets[index] for index in indices]
                data = np.zeros((len(indices), max(counts), self.tokens.shape[1]), dtype=np.float32)
                for position, index in enumerate(indices):
                    start, end = self.offsets[index : index + 2]
                    rows = self.tokens[start:end].astype(np.float32)
                    scales = self.scales[start:end].astype(np.float32)
                    data[position, : end - start] = rows * (scales[:, None] / 127)
                data = torch.from_numpy(data).to(queries)
                scores = [(indices[index], score) for index, score in self.encoder.score(queries, data, None)[0]]

                # Resolve internal slots through the embeddings search paths
                if embeddings.database:
                    scores = [(row["id"], row["score"]) for row in embeddings.database.search("", [scores])]
                else:
                    scores = [(embeddings.ids[index], score) for index, score in scores if embeddings.ids[index] is not None]

                results = sorted(results + scores, key=lambda row: -row[1])[:limit]

        return results

    def save(self, path):
        """
        Saves contiguous arrays to one safetensors file.

        Args:
            path: output file path
        """

        safetensors.numpy.save_file(
            {
                "tokens": np.ascontiguousarray(self.tokens),
                "scales": np.ascontiguousarray(self.scales),
                "offsets": np.ascontiguousarray(self.offsets),
            },
            path,
        )

    def load(self, path):
        """
        Loads the arrays in full for subsequent upserts and deletes.

        Args:
            path: input file path
        """

        with safetensors.safe_open(path, framework="np") as source:
            self.tokens = source.get_tensor("tokens")
            self.scales = source.get_tensor("scales")
            self.offsets = source.get_tensor("offsets")
            self.buffers.clear()

    def close(self):
        """
        Releases encoder, arrays and the owning embeddings reference.
        """

        self.embeddings, self.encoder = None, None

        # Release published arrays and their allocated buffers
        self.tokens, self.scales, self.offsets = None, None, None
        self.buffers.clear()

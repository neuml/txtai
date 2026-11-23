"""
Multi-vector module tests
"""

import gc
import os
import tempfile
import unittest
import weakref

from unittest.mock import patch

import numpy as np
import torch

from txtai.embeddings import Embeddings
from txtai.pipeline import LateEncoder


class TestMultiVector(unittest.TestCase):
    """
    Multi-vector tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

    def testMultiVectorArchive(self):
        """
        Test saving and loading a multi-vector index archive
        """

        # Index data without content
        with tempfile.TemporaryDirectory() as directory, self.multivector(False) as embeddings:
            embeddings.index(self.data[:2])
            original = embeddings.multivector.score("virus cases", [0, 1])

            # Save to an archive, load it and check the scores are unchanged
            archive = os.path.join(directory, "index.tar.gz")
            embeddings.save(archive)
            embeddings.load(archive)
            self.assertEqual(embeddings.multivector.score("virus cases", [0, 1]), original)
            self.assertEqual(len(embeddings.multivector.offsets), 3)

    def testMultiVectorClose(self):
        """
        Test close releases multi-vector arrays and references
        """

        # Index data and close the index
        embeddings = self.multivector()
        embeddings.index(self.data[:1])
        store = embeddings.multivector
        embeddings.close()

        # Every array and reference is released
        self.assertIsNone(embeddings.multivector)
        self.assertIsNone(store.encoder)
        self.assertIsNone(store.embeddings)
        self.assertIsNone(store.tokens)
        self.assertIsNone(store.scales)
        self.assertIsNone(store.offsets)

    def testMultiVectorCloud(self):
        """
        Test saving and loading a multi-vector index with cloud storage
        """

        with tempfile.TemporaryDirectory() as directory, self.multivector() as embeddings:
            # Index data and generate scores
            embeddings.index(self.data[:2])
            original = embeddings.multivector.score("virus cases", ["0", "1"])

            # Save and load through a mock cloud provider
            with patch("txtai.embeddings.base.CloudFactory.create") as create:
                cloud = create.return_value
                cloud.load.return_value = directory
                embeddings.save(directory, cloud={"provider": "mock"})
                cloud.save.assert_called_once_with(directory)
                self.assertTrue(os.path.isfile(os.path.join(directory, "multivector")))
                embeddings.load("remote-index", cloud={"provider": "mock"})
                cloud.load.assert_called_once_with("remote-index")

            # Scores and rows are unchanged
            self.assertEqual(embeddings.multivector.score("virus cases", ["0", "1"]), original)
            self.assertEqual(len(embeddings.multivector.offsets), 3)

    def testMultiVectorDefaultOff(self):
        """
        Test multi-vector is off by default and adds no file
        """

        with tempfile.TemporaryDirectory() as directory:
            with Embeddings(keyword=True, content=True) as embeddings:
                # No multi-vector index before or after indexing
                self.assertIsNone(embeddings.multivector)
                embeddings.index(self.data[:2])
                self.assertIsNone(embeddings.multivector)

                # Save and load writes the original file list
                embeddings.save(directory)
                self.assertEqual(sorted(os.listdir(directory)), ["config.json", "documents", "scoring", "scoring.terms"])
                embeddings.load(directory)
                self.assertIsNone(embeddings.multivector)

    def testMultiVectorDelete(self):
        """
        Test multi-vector delete keeps remaining scores and empty slots
        """

        for content in [True, False]:
            with self.multivector(content) as embeddings:
                # Index data and generate scores
                embeddings.index(self.data)
                store = embeddings.multivector
                ids = [str(x) if content else x for x in range(len(self.data))]
                before = dict(store.score("lottery ticket", ids))

                # First, last, adjacent, duplicate and out-of-order internal ids
                store.delete([5, 2, 0, 3, 2])
                # Repeated deletes of already-empty slots must preserve surviving rows
                store.delete([2, 0, 5])
                remaining = store.score("lottery ticket", ids)
                self.assertEqual({uid for uid, _ in remaining}, {ids[1], ids[4]})
                for uid, score in remaining:
                    self.assertAlmostEqual(score, before[uid], places=6)
                self.assertEqual(len(store.offsets), len(self.data) + 1)
                self.assertEqual(np.diff(store.offsets)[[0, 2, 3, 5]].tolist(), [0, 0, 0, 0])

                # The Embeddings delete hook resolves external ids before compaction
                embeddings.delete([ids[1]])
                remaining = store.score("lottery ticket", ids)
                self.assertEqual([uid for uid, _ in remaining], [ids[4]])
                self.assertAlmostEqual(remaining[0][1], before[ids[4]], places=6)

    def testMultiVectorDisabled(self):
        """
        Test falsy configuration disables multi-vector indexing and saving
        """

        # Each falsy value disables multi-vector indexing and the saved file
        for config in [False, None, 0]:
            with tempfile.TemporaryDirectory() as directory, Embeddings(keyword=True, content=True, multivector=config) as embeddings:
                embeddings.index(self.data[:2])
                self.assertIsNone(embeddings.multivector)
                embeddings.save(directory)
                self.assertFalse(os.path.exists(os.path.join(directory, "multivector")))
                embeddings.load(directory)
                self.assertIsNone(embeddings.multivector)

    def testMultiVectorDtype(self):
        """
        Test candidate tensors match the query dtype and device
        """

        with self.multivector() as embeddings:
            # Index data and encode a float64 query
            embeddings.index(self.data[:2])
            store = embeddings.multivector
            query = store.encoder.encode(["virus cases"], "query").to(dtype=torch.float64, device="cpu")

            # Candidates are converted to the query dtype and device
            with patch.object(store.encoder, "encode", return_value=query), patch.object(store.encoder, "score", wraps=store.encoder.score) as score:
                result = store.score("virus cases", ["0", "1"])
                self.assertEqual(len(result), 2)
                queries, data, _ = score.call_args.args
                self.assertEqual(data.dtype, queries.dtype)
                self.assertEqual(data.device, queries.device)
                self.assertEqual(data.dtype, torch.float64)
                self.assertEqual(data.device.type, "cpu")

    def testMultiVectorIndex(self):
        """
        Test multi-vector scoring of only the supplied search ids
        """

        for content in [True, False]:
            with self.multivector(content) as embeddings:
                # Index data and search
                embeddings.index(self.data)
                store = embeddings.multivector
                results = embeddings.search("lottery ticket", 3)
                ids = [row["id"] if content else row[0] for row in results]

                # Score the search ids with one encoder call
                with patch.object(store.encoder, "score", wraps=store.encoder.score) as score:
                    ranked = store.score("lottery ticket", ids)
                    self.assertEqual(score.call_count, 1)
                self.assertEqual({uid for uid, _ in ranked}, set(ids))
                self.assertEqual(ranked[0][0], "4" if content else 4)
                self.assertEqual(ranked, sorted(ranked, key=lambda row: -row[1]))

                # Limits, empty and unknown ids
                self.assertEqual(store.score("lottery ticket", ids, 1), ranked[:1])
                self.assertEqual(store.score("lottery ticket", []), [])
                self.assertEqual(store.score("lottery ticket", ["unknown"]), [])
                self.assertEqual(len(store.offsets), len(self.data) + 1)

    def testMultiVectorNoneText(self):
        """
        Test metadata without text consumes no index id
        """

        with self.multivector(batch=2, columns={"text": "value"}) as embeddings:
            # Index a metadata row with no text between two text rows
            embeddings.index([("a", {"value": self.data[0]}), ("metadata", {"value": None}), ("b", {"value": self.data[4]})])
            store = embeddings.multivector
            self.assertEqual(len(store.offsets), 3)
            self.assertEqual(embeddings.database.ids(["a", "b"]), [(0, "a"), (1, "b")])
            self.assertEqual(store.score("lottery ticket", ["a", "b"])[0][0], "b")

            # Delete the metadata row and a text row
            embeddings.delete(["metadata", "a"])
            self.assertEqual(store.offsets[0], store.offsets[1])
            self.assertEqual([uid for uid, _ in store.score("lottery ticket", ["a", "b"])], ["b"])

    def testMultiVectorObjects(self):
        """
        Test non-text objects get empty multi-vector rows
        """

        def transform(data):
            return np.array([[len(value), 1] for value in data], dtype=np.float32)

        # Index metadata, an object and tokens with an external transform
        with Embeddings(
            method="external",
            transform=transform,
            backend="numpy",
            content=True,
            batch=1,
            multivector={"path": "neuml/colbert-bert-tiny", "gpu": False},
        ) as embeddings:
            embeddings.index([("meta", {"label": "x"}), ("object", {"object": b"bytes"}), ("text", ["lottery", "ticket"])])
            store = embeddings.multivector
            self.assertEqual(store.offsets.dtype, np.int64)
            self.assertEqual(store.offsets[:2].tolist(), [0, 0])
            self.assertEqual(len(store.offsets), 3)
            self.assertEqual(store.score("lottery", ["object"]), [])
            self.assertEqual(store.score("lottery", ["object", "text"])[0][0], "text")

            # Reindex an object only
            embeddings.index([("object", {"object": b"bytes"})])
            self.assertEqual(embeddings.multivector.offsets.tolist(), [0, 0])
            self.assertEqual(embeddings.multivector.score("lottery", ["object"]), [])

    def testMultiVectorOwnership(self):
        """
        Test upsert only writes to arrays the store allocated
        """

        with self.multivector() as embeddings:
            # Index data and upsert twice, the second upsert reuses earlier rows in place
            embeddings.index([(str(x), text) for x, text in enumerate(self.data)])
            embeddings.upsert([("new1", self.data[0])])
            store = embeddings.multivector
            before = store.tokens
            embeddings.upsert([("new2", self.data[1])])
            self.assertTrue(np.shares_memory(before, store.tokens))

            # Assign a view of a larger outside array
            rows = len(store.tokens)
            outside = np.full((2 * rows + 64, store.tokens.shape[1]), 5, dtype=np.int8)
            outside[:rows] = -store.tokens
            store.tokens = outside[:rows]
            expected = outside.copy()

            # Upsert keeps the assigned rows and never writes to or shares the outside array
            embeddings.upsert([("new3", self.data[2])])
            np.testing.assert_array_equal(outside, expected)
            np.testing.assert_array_equal(store.tokens[:rows], expected[:rows])
            self.assertEqual(len(store.tokens), store.offsets[-1])
            self.assertFalse(np.shares_memory(store.tokens, outside))

            # Upsert rejects stored tokens with a different width
            store.tokens = np.zeros((len(store.tokens), 1), dtype=np.int8)
            with self.assertRaises(ValueError):
                embeddings.upsert([("new4", self.data[3])])

    def testMultiVectorReindex(self):
        """
        Test reindex rebuilds or disables multi-vector rows
        """

        with self.multivector() as embeddings:
            # Index data, delete a row and reindex
            embeddings.index([("a", self.data[0]), ("b", self.data[4])])
            embeddings.delete(["a"])
            previous = embeddings.multivector
            config = embeddings.config.copy()
            embeddings.reindex(config)
            self.assertIsNone(previous.encoder)
            self.assertEqual(len(embeddings.multivector.offsets), 2)
            self.assertEqual(embeddings.database.ids(["b"]), [(0, "b")])
            self.assertEqual(embeddings.multivector.score("lottery ticket", ["a", "b"])[0][0], "b")

            # Reindex without the multi-vector configuration
            config.pop("multivector")
            embeddings.reindex(config)
            self.assertIsNone(embeddings.multivector)

    def testMultiVectorRelease(self):
        """
        Test delete and load release replaced multi-vector arrays
        """

        with tempfile.TemporaryDirectory() as directory, self.multivector() as embeddings:
            # Index data and keep references to the stored arrays
            embeddings.index([(str(x), text) for x, text in enumerate(self.data)])
            store = embeddings.multivector
            offsets = store.offsets
            expected = offsets.copy()
            tokens, scales = weakref.ref(store.tokens), weakref.ref(store.scales)

            # Delete keeps the earlier offsets and releases the replaced arrays
            embeddings.delete(["1"])
            gc.collect()
            np.testing.assert_array_equal(offsets, expected)
            self.assertIsNone(tokens())
            self.assertIsNone(scales())

            # Load releases the replaced arrays
            embeddings.upsert([("new1", self.data[1])])
            embeddings.save(directory)
            tokens, scales = weakref.ref(store.tokens), weakref.ref(store.scales)
            store.load(os.path.join(directory, "multivector"))
            gc.collect()
            self.assertIsNone(tokens())
            self.assertIsNone(scales())

    def testMultiVectorSave(self):
        """
        Test multi-vector save, load and upsert after load
        """

        with tempfile.TemporaryDirectory() as directory, self.multivector() as embeddings:
            # Index data, save and load
            embeddings.index(self.data[:2])
            original = embeddings.multivector.score("virus cases", ["0", "1"])
            embeddings.save(directory)
            self.assertTrue(os.path.isfile(os.path.join(directory, "multivector")))
            embeddings.load(directory)
            self.assertEqual(embeddings.multivector.score("virus cases", ["0", "1"]), original)

            # Upsert after load
            embeddings.upsert([("0", self.data[4])])
            self.assertEqual(embeddings.multivector.score("lottery ticket", ["0", "1"])[0][0], "0")

    def testMultiVectorScore(self):
        """
        Test stored scores against freshly encoded candidates
        """

        for content in [True, False]:
            with self.multivector(content) as embeddings:
                # Index data and encode the same data
                embeddings.index(self.data)
                store = embeddings.multivector
                ids = [str(x) if content else x for x in range(len(self.data))]
                rows = store.encoder.encode(self.data, "data")

                # Stored scores match scores of the freshly encoded rows
                for query in ["lottery ticket", "ice shelf", "bear attack"]:
                    queries = store.encoder.encode([query], "query")
                    fresh = [(ids[index], score) for index, score in store.encoder.score(queries, rows, None)[0]]
                    stored = store.score(query, ids)
                    self.assertEqual([uid for uid, _ in stored], [uid for uid, _ in fresh])
                    for (uid, score), (expectedid, expected) in zip(stored, fresh):
                        self.assertEqual(uid, expectedid)
                        self.assertLessEqual(abs(score - expected), 0.02)

    def testMultiVectorShortcut(self):
        """
        Test shortcut configuration uses LateEncoder defaults
        """

        # True and an empty dict create the encoder with no arguments
        for config in [True, {}]:
            with patch("txtai.embeddings.index.multivector.LateEncoder") as encoder:
                encoder.return_value = LateEncoder("neuml/colbert-bert-tiny", gpu=False)
                with Embeddings(keyword=True, multivector=config) as embeddings:
                    embeddings.index(self.data[:1])
                    encoder.assert_called_once_with()
                    self.assertEqual(embeddings.config["multivector"], {})
                    self.assertEqual(len(embeddings.multivector.offsets), 2)

    def testMultiVectorStorage(self):
        """
        Test quantized multi-vector storage and padding removal
        """

        # Index two rows with fixed encoded values
        rows = torch.tensor([[[1.0, 0.5, -1.0], [0.0, 0.0, 0.0]], [[0.25, -0.5, 0.125], [0.0, 0.0, 0.0]]], dtype=torch.float32)
        with self.multivector() as embeddings, patch("txtai.embeddings.index.multivector.LateEncoder.encode", return_value=rows):
            embeddings.index(self.data[:2])
            store = embeddings.multivector

            # Stored arrays equal the expected quantized values without padding
            tokens = np.array([[127, 64, -127], [64, -127, 32]], dtype=np.int8)
            scales = np.array([1.0, 0.5], dtype=np.float16)
            offsets = np.array([0, 1, 2], dtype=np.int64)
            self.assertEqual(store.tokens.dtype, tokens.dtype)
            self.assertEqual(store.scales.dtype, scales.dtype)
            self.assertEqual(store.offsets.dtype, offsets.dtype)
            np.testing.assert_array_equal(store.tokens, tokens)
            np.testing.assert_array_equal(store.scales, scales)
            np.testing.assert_array_equal(store.offsets, offsets)

    def testMultiVectorUpsert(self):
        """
        Test multi-vector upsert replaces and appends rows
        """

        for content in [True, False]:
            with self.multivector(content, batch=1) as embeddings:
                # Upsert a replaced row and a new row
                embeddings.upsert([("a", self.data[0]), ("b", self.data[1])])
                embeddings.upsert([("a", self.data[4]), ("c", self.data[3])])
                store = embeddings.multivector
                self.assertEqual(len(store.offsets), 5)
                self.assertEqual(store.offsets[0], store.offsets[1])
                self.assertGreater(store.offsets[3], store.offsets[2])
                self.assertEqual(store.score("lottery ticket", ["a", "b", "c"])[0][0], "a")
                self.assertEqual(embeddings.count(), 3)

                # Delete an existing and a missing id
                embeddings.delete(["b", "missing"])
                self.assertEqual(store.offsets[1], store.offsets[2])
                self.assertEqual({uid for uid, _ in store.score("lottery ticket", ["a", "b", "c"])}, {"a", "c"})

    def multivector(self, content=True, **kwargs):
        """
        Creates an embeddings instance with a multi-vector index.

        Args:
            content: enable content storage, defaults to True
            kwargs: additional configuration

        Returns:
            Embeddings
        """

        return Embeddings(
            path="sentence-transformers/nli-mpnet-base-v2",
            backend="numpy",
            gpu=False,
            content=content,
            multivector={"path": "neuml/colbert-bert-tiny", "gpu": False},
            **kwargs,
        )

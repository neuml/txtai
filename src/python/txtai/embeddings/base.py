"""
Embeddings module
"""

import pickle
import os
import shutil

import numpy as np

from ..ann import ANNFactory
from ..scoring import ScoringFactory
from ..vectors import VectorsFactory

from .reducer import Reducer


class Embeddings:
    """
    Model that builds sentence embeddings from a list of tokens.

    Optional scoring method can be created to weigh tokens when creating embeddings. Averaging used if no scoring method provided.

    The model also applies principal component analysis using a LSA model. This reduces the noise of common but less
    relevant terms.
    """

    # pylint: disable = W0231
    def __init__(self, config=None):
        """
        Creates a new Embeddings model.

        Args:
            config: embeddings configuration
        """

        # Configuration
        self.config = config

        # Embeddings model
        self.embeddings = None

        if self.config and self.config.get("method") != "transformers":
            # Dimensionality reduction model
            self.reducer = None

            # Embedding scoring method - weighs each word in a sentence
            self.scoring = ScoringFactory.create(self.config["scoring"]) if self.config and self.config.get("scoring") else None
        else:
            self.reducer, self.scoring = None, None

        # Sentence vectors model
        self.model = self.loadVectors() if self.config else None

    def score(self, documents):
        """
        Builds a scoring index.

        Args:
            documents: list of (id, text|tokens, tags)
        """

        if self.scoring:
            # Build scoring index over documents
            self.scoring.index(documents)

    def index(self, documents):
        """
        Builds an embeddings index. This method overwrites an existing index.

        Args:
            documents: list of (id, text|tokens, tags)
        """

        # Transform documents to embeddings vectors
        ids, dimensions, embeddings = self.vectors(documents)

        # Build LSA model (if enabled). Remove principal components from embeddings.
        if self.config.get("pca"):
            self.reducer = Reducer(embeddings, self.config["pca"])
            self.reducer(embeddings)

        # Normalize embeddings
        self.normalize(embeddings)

        # Save embeddings metadata
        self.config["ids"] = ids
        self.config["dimensions"] = dimensions

        # Create embeddings index
        self.embeddings = ANNFactory.create(self.config)

        # Build the index
        self.embeddings.index(embeddings)

    def upsert(self, documents):
        """
        Runs an embeddings index upsert operation. If the index exists, new
        data is appended to the index, existing data is updated. If the index
        doesn't exist, this method runs a standard index operation.

        Args:
            documents: list of (id, text|tokens, tags)
        """

        # Run standard insert if index doesn't exist
        if not self.embeddings:
            self.index(documents)
            return

        # Transform documents to embeddings vectors
        ids, _, embeddings = self.vectors(documents)

        # Normalize embeddings
        self.normalize(embeddings)

        # Delete existing elements
        self.delete(ids)

        # Append elements the index
        self.embeddings.append(embeddings)

        # Save embeddings metadata
        self.config["ids"] = self.config["ids"] + ids

    def delete(self, ids):
        """
        Deletes from an embeddings index. Returns list of ids deleted.

        Args:
            ids: list of ids to delete

        Returns:
            ids deleted
        """

        # List of internal indices for each candidate id to delete
        indices = []

        # List of deleted ids
        deletes = []

        # Get handle to config ids
        cids = self.config["ids"]

        # Find existing ids
        for uid in ids:
            indices.extend([index for index, value in enumerate(cids) if uid == value])

        # Delete any found from config ids and embeddings
        if indices:
            # Clear config ids
            for index in indices:
                deletes.append(cids[index])
                cids[index] = None

            # Delete ids from index
            self.embeddings.delete(indices)

        return deletes

    def transform(self, document):
        """
        Transforms document into an embeddings vector. Document text will be tokenized if not pre-tokenized.

        Args:
            document: (id, text|tokens, tags)

        Returns:
            embeddings vector
        """

        # Convert document into sentence embedding
        embedding = self.model.transform(document)

        # Reduce the dimensionality of the embeddings. Scale the embeddings using this
        # model to reduce the noise of common but less relevant terms.
        if self.reducer:
            self.reducer(embedding)

        # Normalize embeddings
        self.normalize(embedding)

        return embedding

    def batchtransform(self, documents):
        """
        Transforms documents into embeddings vectors. Document text will be tokenized if not pre-tokenized.

        Args:
            documents: list of (id, text|tokens, tags)

        Returns:
            embeddings vectors
        """

        return [self.transform(document) for document in documents]

    def count(self):
        """
        Total number of elements in this embeddings index.

        Returns:
            number of elements in embeddings index
        """

        return self.embeddings.count()

    def search(self, query, limit=3):
        """
        Finds documents in the embeddings model most similar to the input query. Returns
        a list of (id, score) sorted by highest score, where id is the document id in
        the embeddings model.

        Args:
            query: query text|tokens
            limit: maximum results

        Returns:
            list of (id, score)
        """

        return self.batchsearch([query], limit)[0]

    def batchsearch(self, queries, limit=3):
        """
        Finds documents in the embeddings model most similar to the input queries. Returns
        a list of (id, score) sorted by highest score per query, where id is the document id
        in the embeddings model.

        Args:
            queries: queries text|tokens
            limit: maximum results

        Returns:
            list of (id, score) per query
        """

        # Convert queries to embedding vectors
        embeddings = np.array([self.transform((None, query, None)) for query in queries])

        # Search embeddings index
        results = self.embeddings.search(embeddings, limit)

        # Map ids if id mapping available
        lookup = self.config.get("ids")
        if lookup:
            results = [[(lookup[i], score) for i, score in r] for r in results]

        return results

    def similarity(self, query, texts):
        """
        Computes the similarity between query and list of text. Returns a list of
        (id, score) sorted by highest score, where id is the index in texts.

        Args:
            query: query text|tokens
            texts: list of text|tokens

        Returns:
            list of (id, score)
        """

        return self.batchsimilarity([query], texts)[0]

    def batchsimilarity(self, queries, texts):
        """
        Computes the similarity between list of queries and list of text. Returns a list
        of (id, score) sorted by highest score per query, where id is the index in texts.

        Args:
            queries: queries text|tokens
            texts: list of text|tokens

        Returns:
            list of (id, score) per query
        """

        # Convert queries to embedding vectors
        queries = np.array([self.transform((None, query, None)) for query in queries])
        texts = np.array([self.transform((None, text, None)) for text in texts])

        # Dot product on normalized vectors is equal to cosine similarity
        scores = np.dot(queries, texts.T).tolist()

        # Add index id and sort desc based on score
        return [sorted(enumerate(score), key=lambda x: x[1], reverse=True) for score in scores]

    def load(self, path):
        """
        Loads a pre-trained model.

        Models have the following files:
            config - configuration
            embeddings - sentence embeddings index
            lsa - LSA model, used to remove the principal component(s)
            scoring - scoring model used to weigh word vectors
            vectors - vectors model

        Args:
            path: input directory path
        """

        # Index configuration
        with open("%s/config" % path, "rb") as handle:
            self.config = pickle.load(handle)

            # Build full path to embedding vectors file
            if self.config.get("storevectors"):
                self.config["path"] = os.path.join(path, self.config["path"])

        # Sentence embeddings index
        self.embeddings = ANNFactory.create(self.config)
        self.embeddings.load("%s/embeddings" % path)

        # Dimensionality reduction
        if self.config.get("pca"):
            with open("%s/lsa" % path, "rb") as handle:
                self.reducer = Reducer()
                self.reducer.load(path)

        # Embedding scoring
        if self.config.get("scoring"):
            self.scoring = ScoringFactory.create(self.config["scoring"])
            self.scoring.load(path)

        # Sentence vectors model - transforms text into sentence embeddings
        self.model = self.loadVectors()

    def exists(self, path):
        """
        Checks if an index exists at path.

        Args:
            path: input directory path

        Returns:
            true if index exists, false otherwise
        """

        return os.path.exists("%s/config" % path) and os.path.exists("%s/embeddings" % path)

    def save(self, path):
        """
        Saves a model.

        Args:
            path: output directory path
        """

        if self.config:
            # Create output directory, if necessary
            os.makedirs(path, exist_ok=True)

            # Copy vectors file
            if self.config.get("storevectors"):
                shutil.copyfile(self.config["path"], os.path.join(path, os.path.basename(self.config["path"])))

                self.config["path"] = os.path.basename(self.config["path"])

            # Write index configuration
            with open("%s/config" % path, "wb") as handle:
                pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Write sentence embeddings index
            self.embeddings.save("%s/embeddings" % path)

            # Save dimensionality reduction
            if self.reducer:
                self.reducer.save(path)

            # Save embedding scoring
            if self.scoring:
                self.scoring.save(path)

    def loadVectors(self):
        """
        Loads a vector model set in config.

        Returns:
            vector model
        """

        return VectorsFactory.create(self.config, self.scoring)

    def vectors(self, documents):
        """
        Transforms documents into embeddings vectors.

        Args:
            documents: list of (id, text|tokens, tags)

        Returns:
            tuple of document ids, dimensions and embeddings
        """

        # Transform documents to embeddings vectors
        ids, dimensions, stream = self.model.index(documents)

        # Load streamed embeddings back to memory
        embeddings = np.empty((len(ids), dimensions), dtype=np.float32)
        with open(stream, "rb") as queue:
            for x in range(embeddings.shape[0]):
                embeddings[x] = pickle.load(queue)

        # Remove temporary file
        os.remove(stream)

        return (ids, dimensions, embeddings)

    def normalize(self, embeddings):
        """
        Normalizes embeddings using L2 normalization. Operation applied directly on array.

        Args:
            embeddings: input embeddings matrix
        """

        # Calculation is different for matrices vs vectors
        if len(embeddings.shape) > 1:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        else:
            embeddings /= np.linalg.norm(embeddings)

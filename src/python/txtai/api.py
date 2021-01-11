"""
API module
"""

import os
import pickle
import tempfile

from typing import List

import yaml

from fastapi import Body, FastAPI, Request

from .embeddings import Embeddings
from .extractor import Extractor
from .pipeline import Labels, Similarity

# API instance
app = FastAPI()

# Global API instance
INSTANCE = None

class API(object):
    """
    Base API template. Downstream applications can extend this base template to add custom search functionality.
    """

    def __init__(self, config):
        """
        Creates an embeddings index instance that is called by FastAPI.

        Args:
            config: index configuration
        """

        # Initialize member variables
        self.config, self.documents, self.batch = config, None, 0
        self.embeddings, self.extractor, self.labels, self.similar = None, None, None, None

        # Create/load embeddings index depending on writable flag
        if self.config.get("writable"):
            self.embeddings = Embeddings(self.config["embeddings"])
        elif self.config.get("path"):
            self.embeddings = Embeddings()
            self.embeddings.load(self.config["path"])

        # Create extractor instance
        if "extractor" in self.config:
            # Extractor settings
            extractor = self.config["extractor"] if self.config["extractor"] else {}

            self.extractor = Extractor(self.embeddings, extractor.get("path"), extractor.get("quantize"),
                                       extractor.get("gpu"))

        # Create labels instance
        if "labels" in self.config:
            labels = self.config["labels"] if self.config["labels"] else {}

            self.labels = Labels(labels.get("path"), labels.get("quantize"), labels.get("gpu"))

        # Creates similarity instance
        if "similarity" in self.config:
            similarity = self.config["similarity"] if self.config["similarity"] else {}

            # Share model with labels if separate model not specified
            if "path" not in similarity and self.labels:
                self.similar = Similarity(model=self.labels)
            else:
                self.similar = Similarity(similarity.get("path"), similarity.get("quantize"), similarity.get("gpu"))

    def limit(self, limit):
        """
        Parses the number of results to return from the request. Allows range of 1-250, with a default of 10.

        Args:
            limit: limit parameter

        Returns:
            bounded limit
        """

        # Return between 1 and 250 results, defaults to 10
        return max(1, min(250, int(limit) if limit else 10))

    def search(self, query, request):
        """
        Finds documents in the embeddings model most similar to the input query. Returns
        a list of (id, score) sorted by highest score, where id is the document id in
        the embeddings model.

        Downstream applications can override this method to provide enriched search results.

        Args:
            query: query text
            request: FastAPI request

        Returns:
            list of (id, score)
        """

        if self.embeddings:
            return self.embeddings.search(query, self.limit(request.query_params.get("limit")))

        return None

    def batchsearch(self, queries, limit):
        """
        Finds documents in the embeddings model most similar to the input queries. Returns
        a list of (id, score) sorted by highest score per query, where id is the document id
        in the embeddings model.

        Args:
            queries: queries text
            limit: maximum results

        Returns:
            list of (id, score) per query
        """

        if self.embeddings:
            return self.embeddings.batchsearch(queries, self.limit(limit))

        return None

    def add(self, documents):
        """
        Adds a batch of documents for indexing.

        Args:
            documents: list of {id: value, text: value}
        """

        # Only add batch if index is marked writable
        if self.embeddings and self.config.get("writable"):
            # Create documents file if not already open
            if not self.documents:
                self.documents = tempfile.NamedTemporaryFile(mode="wb", suffix=".docs", delete=False)

            # Add batch
            pickle.dump([(document["id"], document["text"], None) for document in documents], self.documents)
            self.batch += 1

    def stream(self):
        """
        Generator that streams documents previously queued with add.
        """

        # Open stream file
        with open(self.documents.name, "rb") as queue:
            # Read each batch
            for _ in range(self.batch):
                documents = pickle.load(queue)

                # Yield each document
                for document in documents:
                    yield document

    def index(self):
        """
        Builds an embeddings index for previously batched documents. No further documents can be added
        after this call.

        Downstream applications can override this method to also store full documents in an external system.
        """

        if self.embeddings and self.config.get("writable") and self.documents:
            # Close streaming file
            self.documents.close()

            # Build scoring index if scoring method provided
            if self.config.get("scoring"):
                embeddings.score(self.stream())

            # Build embeddings index
            self.embeddings.index(self.stream())

            # Save index
            self.embeddings.save(self.config["path"])

            # Cleanup stream file
            os.remove(self.documents.name)

            # Reset document parameters
            self.documents = None
            self.batch = 0

    def similarity(self, query, texts):
        """
        Computes the similarity between query and list of strings. Returns a list of
        (id, score) sorted by highest score, where id is the index in texts.

        Args:
            query: query text
            texts: list of text

        Returns:
            list of (id, score)
        """

        # Use similarity instance if available otherwise fall back to embeddings model
        if self.similar:
            return [(uid, float(text)) for uid, text in self.similar(query, texts)]
        if self.embeddings:
            return [(uid, float(text)) for uid, text in self.embeddings.similarity(query, texts)]

        return None

    def batchsimilarity(self, queries, texts):
        """
        Computes the similarity between list of queries and list of strings. Returns a list
        of (id, score) sorted by highest score per query, where id is the index in texts.

        Args:
            queries: queries text
            texts: list of text

        Returns:
            list of (id, score) per query
        """

        # Use similarity instance if available otherwise fall back to embeddings model
        if self.similar:
            return [[(uid, float(text)) for uid, text in r] for r in self.similar(queries, texts)]
        if self.embeddings:
            return [[(uid, float(text)) for uid, text in r] for r in self.embeddings.batchsimilarity(queries, texts)]

        return None

    def transform(self, texts):
        """
        Transforms list of texts into embeddings arrays.

        Args:
            texts: list of strings

        Returns:
            embeddings arrays
        """

        if self.embeddings:
            return [[float(x) for x in self.embeddings.transform((None, text, None))] for text in texts]

        return None

    def extract(self, queue, texts):
        """
        Extracts answers to input questions.

        Args:
            queue: list of {name: value, query: value, question: value, snippet: value}
            texts: list of strings

        Returns:
            list of (name, answer)
        """

        if self.extractor:
            # Convert queue to tuples
            queue = [(x["name"], x["query"], x.get("question"), x.get("snippet")) for x in queue]
            return self.extractor(queue, texts)

        return None

    def label(self, text, labels):
        """
        Applies a zero shot classifier to text using a list of labels. Returns a list of
        (id, score) sorted by highest score, where id is the index in labels.

        Args:
            text: text|list
            labels: list of labels

        Returns:
            list of (id, score) per text element
        """

        if self.labels:
            return self.labels(text, labels)

        return None

class Factory(object):
    """
    API factory. Creates new API instances.
    """

    @staticmethod
    def get(atype):
        """
        Gets a new instance of atype.

        Args:
            atype: API instance class

        Returns:
            instance of atype
        """

        parts = atype.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

        return m

@app.on_event("startup")
def start():
    """
    FastAPI startup event. Pre-loads embeddings index.
    """

    # pylint: disable=W0603
    global INSTANCE

    # Load YAML settings
    with open(os.getenv("CONFIG"), "r") as f:
        # Read configuration
        config = yaml.safe_load(f)

    # Instantiate API instance
    api = os.getenv("API_CLASS")
    INSTANCE = Factory.get(api)(config) if api else API(config)

@app.get("/search")
def search(query: str, request: Request):
    """
    Finds documents in the embeddings model most similar to the input query. Returns
    a list of (id, score) sorted by highest score, where id is the document id in
    the embeddings model.

    Args:
        query: query text
        request: FastAPI request

    Returns:
        list of (id, score)
    """

    return INSTANCE.search(query, request)

@app.post("/batchsearch")
def batchsearch(queries: List[str] = Body(...), limit: int = Body(...)):
    """
    Finds documents in the embeddings model most similar to the input queries. Returns
    a list of (id, score) sorted by highest score per query, where id is the document id
    in the embeddings model.

    Args:
        queries: queries text
        limit: maximum results

    Returns:
        list of (id, score) per query
    """

    return INSTANCE.batchsearch(queries, limit)

@app.post("/add")
def add(documents: List[dict] = Body(...)):
    """
    Adds a batch of documents for indexing.

    Args:
        documents: list of {id: value, text: value}
    """

    INSTANCE.add(documents)

@app.get("/index")
def index():
    """
    Builds an embeddings index for previously batched documents. No further documents can be added
    after this call.
    """

    INSTANCE.index()

@app.post("/similarity")
def similarity(query: str = Body(...), texts: List[str] = Body(...)):
    """
    Computes the similarity between query and list of strings. Returns a list of
    (id, score) sorted by highest score, where id is the index in texts.

    Args:
        query: query text
        texts: list of text

    Returns:
        list of (id, score)
    """

    return INSTANCE.similarity(query, texts)

@app.post("/batchsimilarity")
def batchsimilarity(queries: List[str] = Body(...), texts: List[str] = Body(...)):
    """
    Computes the similarity between list of queries and list of strings. Returns a list
    of (id, score) sorted by highest score per query, where id is the index in texts.

    Args:
        queries: queries text
        texts: list of text

    Returns:
        list of (id, score) per query
    """

    return INSTANCE.batchsimilarity(queries, texts)

@app.get("/embeddings")
def embeddings(text: str):
    """
    Transforms text into an embeddings array.

    Args:
        text: input text

    Returns:
        embeddings array
    """

    return INSTANCE.transform([text])[0]

@app.post("/batchembeddings")
def batchembeddings(texts: List[str] = Body(...)):
    """
    Transforms list of text into embeddings arrays.

    Args:
        texts: list of strings

    Returns:
        embeddings arrays
    """

    return INSTANCE.transform(texts)

@app.post("/extract")
def extract(queue: List[dict] = Body(...), texts: List[str] = Body(...)):
    """
    Extracts answers to input questions.

    Args:
        queue: list of {name: value, query: value, question: value, snippet: value)
        texts: list of strings

    Returns:
        list of (name, answer)
    """

    return INSTANCE.extract(queue, texts)

@app.post("/label")
def label(text: str = Body(...), labels: List[str] = Body(...)):
    """
    Applies a zero shot classifier to text using a list of labels. Returns a list of
    (id, score) sorted by highest score, where id is the index in labels.

    Args:
        text: input text
        labels: list of labels

    Returns:
        list of (id, score)
    """

    return INSTANCE.label(text, labels)

@app.post("/batchlabel")
def batchlabel(texts: List[str] = Body(...), labels: List[str] = Body(...)):
    """
    Applies a zero shot classifier to list of text using a list of labels. Returns a list of
    (id, score) sorted by highest score, where id is the index in labels per text element.

    Args:
        texts: list of texts
        labels: list of labels

    Returns:
        list of (id, score) per text element
    """

    return INSTANCE.label(texts, labels)

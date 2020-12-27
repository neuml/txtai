"""
API module
"""

import os

from typing import List

import yaml

from fastapi import Body, FastAPI, Request

from .embeddings import Embeddings
from .extractor import Extractor
from .pipeline import Labels

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
        self.config, self.documents, self.embeddings, self.extractor, self.labels = config, None, None, None, None

        # Create/load embeddings index depending on writable flag
        if self.config.get("writable"):
            self.embeddings = Embeddings(self.config["embeddings"])
        elif self.config.get("path"):
            self.embeddings = Embeddings()
            self.embeddings.load(self.config["path"])

        # Create extractor instance
        if "extractor" in self.config:
            # Extractor settings
            extractor = self.config["extractor"]

            self.extractor = Extractor(self.embeddings, extractor.get("path"), extractor.get("quantize"),
                                       extractor.get("gpu"))

        # Create labels instance
        if "labels" in self.config:
            labels = self.config["labels"]

            self.labels = Labels(labels.get("path"), labels.get("quantize"), labels.get("gpu"))

    def size(self, request):
        """
        Parses the number of results to return from the request. Allows range of 1-250, with a default of 10.

        Args:
            request: FastAPI request

        Returns:
            size
        """

        # Return between 1 and 250 results, defaults to 10
        return max(1, min(250, int(request.query_params["n"]) if "n" in request.query_params else 10))

    def search(self, query, request):
        """
        Runs an embeddings search for query and request. Downstream applications can override this method
        to provide enriched search results.

        Args:
            query: input query
            request: FastAPI request

        Returns:
            list of (uid, score)
        """

        if self.embeddings:
            return self.embeddings.search(query, self.size(request))

        return None

    def add(self, documents):
        """
        Adds a batch of documents for indexing.

        Args:
            documents: list of {id: value, text: value}
        """

        # Only add batch if index is marked writable
        if self.embeddings and self.config.get("writable"):
            # Create current batch if necessary
            if not self.documents:
                self.documents = []

            # Add batch
            self.documents.extend([(document["id"], document["text"], None) for document in documents])

    def index(self):
        """
        Builds an embeddings index. No further documents can be added after this call. Downstream applications can
        override this method to also store full documents in an external system.
        """

        if self.embeddings and self.config.get("writable") and self.documents:
            # Build scoring index if scoring method provided
            if self.config.get("scoring"):
                embeddings.score(self.documents)

            # Build embeddings index
            self.embeddings.index(self.documents)

            # Save index
            self.embeddings.save(self.config["path"])

            # Clear buffer
            self.documents = None

    def similarity(self, search, data):
        """
        Calculates the similarity between text1 and list of elements in text2.

        Args:
            search: text
            data: list of text to compare against

        Returns:
            list of similarity scores
        """

        if self.embeddings:
            return [float(x) for x in self.embeddings.similarity(search, data)]

        return None

    def transform(self, text):
        """
        Transforms text into an embeddings array.

        Args:
            text: input text

        Returns:
            embeddings array
        """

        if self.embeddings:
            return [float(x) for x in self.embeddings.transform((None, text, None))]

        return None

    def extract(self, documents, queue):
        """
        Extracts answers to input questions

        Args:
            documents: list of {id: value, text: value}
            queue: list of {name: value, query: value, question: value, snippet: value)

        Returns:
            extracted answers
        """

        if self.extractor:
            # Convert to a list of (id, text)
            sections = [(document["id"], document["text"]) for document in documents]

            # Convert queue to tuples
            queue = [(x["name"], x["query"], x.get("question"), x.get("snippet")) for x in queue]

            return self.extractor(sections, queue)

        return None

    def label(self, text, labels):
        """
        Applies a zero shot classifier to a text section using a list of labels.

        Args:
            text: input text
            labels: list of labels

        Returns:
            list of (label, score) for section
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
def search(q: str, request: Request):
    """
    Runs an embeddings search.

    Args:
        q: query string
        request: FastAPI request

    Returns:
        query results
    """

    return INSTANCE.search(q, request)

@app.post("/add")
def add(documents: List[dict]):
    """
    Adds a batch of documents for indexing.

    Args:
        documents: list of dicts with each entry containing an id and text element
    """

    INSTANCE.add(documents)

@app.get("/index")
def index():
    """
    Builds an index for previously batched documents.
    """

    INSTANCE.index()

@app.post("/similarity")
def similarity(search: str = Body(None), data: List[str] = Body(None)):
    """
    Calculates the similarity between text1 and list of elements in text2.

    Args:
        search: search text
        data: list of text to compare against

    Returns:
        list of similarity scores
    """

    return INSTANCE.similarity(search, data)

@app.get("/embeddings")
def embeddings(t: str):
    """
    Transforms text into an embeddings array.

    Args:
        t: input text

    Returns:
        embeddings array
    """

    return INSTANCE.transform(t)

@app.post("/extract")
def extract(documents: List[dict], queue: List[dict]):
    """
    Extracts answers to input questions

    Args:
        documents: list of {id: value, text: value}
        queue: list of {name: value, query: value, question: value, snippet: value)

    Returns:
        extracted answers
    """

    return INSTANCE.extract(documents, queue)

@app.post("/label")
def label(text: str = Body(None), labels: List[str] = Body(None)):
    """
    Applies a zero shot classifier to a text section using a list of labels.

    Args:
        text: input text
        labels: list of labels

    Returns:
        list of (label, score) for section
    """

    return INSTANCE.label(text, labels)

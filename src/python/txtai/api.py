"""
API module
"""

import os

from typing import List

import yaml

from fastapi import FastAPI, Query, Request

from .embeddings import Embeddings

# API instance
app = FastAPI()

# Embeddings index instance
INDEX = None

class API(object):
    """
    Base API template. Downstream applications can extend this base template to add custom search functionality.
    """

    def __init__(self, index):
        """
        Creates an embeddings index instance that is called by FastAPI.

        Args:
            index: index configuration
        """

        # Store index settings
        self.index = index

        self.embeddings = Embeddings()
        self.embeddings.load(self.index["path"])

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

        return self.embeddings.search(query, self.size(request))

    def similarity(self, text1, text2):
        """
        Calculates the similarity between text1 and list of elements in text2.

        Args:
            text1: text
            text2: list of text to compare against

        Returns:
            list of similarity scores
        """

        return [float(x) for x in self.embeddings.similarity(text1, text2)]

    def transform(self, text):
        """
        Transforms text into an embeddings array.

        Args:
            text: input text

        Returns:
            embeddings array
        """

        return [float(x) for x in self.embeddings.transform((None, text, None))]

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
        m = __import__( module )
        for comp in parts[1:]:
            m = getattr(m, comp)

        return m

@app.on_event("startup")
def start():
    """
    FastAPI startup event. Pre-loads embeddings index.
    """

    # pylint: disable=W0603
    global INDEX

    # Load YAML settings
    with open(os.getenv("INDEX_SETTINGS"), "r") as f:
        # Read configuration
        index = yaml.safe_load(f)

    # Instantiate API class
    api = os.getenv("API_CLASS")
    INDEX = Factory.get(api)(index) if api else API(index)

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

    return INDEX.search(q, request)

@app.get("/similarity")
def similarity(t1: str, t2: List[str]=Query(None)):
    """
    Calculates the similarity between text1 and list of elements in text2.

    Args:
        t1: text
        t2: list of text to compare against

    Returns:
        list of similarity scores
    """

    return INDEX.similarity(t1, t2)

@app.get("/embeddings")
def embeddings(t: str):
    """
    Transforms text into an embeddings array.

    Args:
        t: input text

    Returns:
        embeddings array
    """

    return INDEX.transform(t)

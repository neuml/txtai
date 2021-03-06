"""
FastAPI application module
"""

import os

from typing import List

import yaml

from fastapi import Body, FastAPI, Request

from .base import API
from .factory import Factory

# API instance
app = FastAPI()

# Global API instance
INSTANCE = None


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
    a list of {id: value, score: value} sorted by highest score, where id is the
    document id in the embeddings model.

    Args:
        query: query text
        request: FastAPI request

    Returns:
        list of {id: value, score: value}
    """

    return INSTANCE.search(query, request)


@app.post("/batchsearch")
def batchsearch(queries: List[str] = Body(...), limit: int = Body(...)):
    """
    Finds documents in the embeddings model most similar to the input queries. Returns
    a list of {id: value, score: value} sorted by highest score per query, where id is
    the document id in the embeddings model.

    Args:
        queries: queries text
        limit: maximum results

    Returns:
        list of {id: value, score: value} per query
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
    Computes the similarity between query and list of text. Returns a list of
    {id: value, score: value} sorted by highest score, where id is the index
    in texts.

    Args:
        query: query text
        texts: list of text

    Returns:
        list of {id: value, score: value}
    """

    return INSTANCE.similarity(query, texts)


@app.post("/batchsimilarity")
def batchsimilarity(queries: List[str] = Body(...), texts: List[str] = Body(...)):
    """
    Computes the similarity between list of queries and list of text. Returns a list
    of {id: value, score: value} sorted by highest score per query, where id is the
    index in texts.

    Args:
        queries: queries text
        texts: list of text

    Returns:
        list of {id: value, score: value} per query
    """

    return INSTANCE.batchsimilarity(queries, texts)


@app.get("/transform")
def transform(text: str):
    """
    Transforms text into an embeddings array.

    Args:
        text: input text

    Returns:
        embeddings array
    """

    return INSTANCE.transform(text)


@app.post("/batchtransform")
def batchtransform(texts: List[str] = Body(...)):
    """
    Transforms list of text into embeddings arrays.

    Args:
        texts: list of text

    Returns:
        embeddings arrays
    """

    return INSTANCE.batchtransform(texts)


@app.post("/extract")
def extract(queue: List[dict] = Body(...), texts: List[str] = Body(...)):
    """
    Extracts answers to input questions.

    Args:
        queue: list of {name: value, query: value, question: value, snippet: value}
        texts: list of text

    Returns:
        list of {name: value, answer: value}
    """

    return INSTANCE.extract(queue, texts)


@app.post("/label")
def label(text: str = Body(...), labels: List[str] = Body(...)):
    """
    Applies a zero shot classifier to text using a list of labels. Returns a list of
    {id: value, score: value} sorted by highest score, where id is the index in labels.

    Args:
        text: input text
        labels: list of labels

    Returns:
        list of {id: value, score: value} per text element
    """

    return INSTANCE.label(text, labels)


@app.post("/batchlabel")
def batchlabel(texts: List[str] = Body(...), labels: List[str] = Body(...)):
    """
    Applies a zero shot classifier to list of text using a list of labels. Returns a list of
    {id: value, score: value} sorted by highest score, where id is the index in labels per
    text element.

    Args:
        texts: list of text
        labels: list of labels

    Returns:
        list of {id: value score: value} per text element
    """

    return INSTANCE.label(texts, labels)

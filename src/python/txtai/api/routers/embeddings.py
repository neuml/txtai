"""
Defines API paths for embeddings endpoints.
"""

from typing import List

from fastapi import APIRouter, Body, HTTPException, Request

from .. import application
from ...app import ReadOnlyError

router = APIRouter()


@router.get("/search")
def search(query: str, request: Request):
    """
    Finds documents most similar to the input query. This method will run either an index search
    or an index + database search depending on if a database is available.

    Args:
        query: input query
        request: FastAPI request

    Returns:
        list of {id: value, score: value} for index search, list of dict for an index + database search
    """

    return application.get().search(query, request=request)


# pylint: disable=W0621
@router.post("/batchsearch")
def batchsearch(
    queries: List[str] = Body(...), limit: int = Body(default=None), weights: float = Body(default=None), index: str = Body(default=None)
):
    """
    Finds documents most similar to the input queries. This method will run either an index search
    or an index + database search depending on if a database is available.

    Args:
        queries: input queries
        limit: maximum results
        weights: hybrid score weights, if applicable
        index: index name, if applicable

    Returns:
        list of {id: value, score: value} per query for index search, list of dict per query for an index + database search
    """

    return application.get().batchsearch(queries, limit, weights, index)


@router.post("/add")
def add(documents: List[dict] = Body(...)):
    """
    Adds a batch of documents for indexing.

    Args:
        documents: list of {id: value, text: value, tags: value}
    """

    try:
        application.get().add(documents)
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.get("/index")
def index():
    """
    Builds an embeddings index for previously batched documents.
    """

    try:
        application.get().index()
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.get("/upsert")
def upsert():
    """
    Runs an embeddings upsert operation for previously batched documents.
    """

    try:
        application.get().upsert()
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.post("/delete")
def delete(ids: List = Body(...)):
    """
    Deletes from an embeddings index. Returns list of ids deleted.

    Args:
        ids: list of ids to delete

    Returns:
        ids deleted
    """

    try:
        return application.get().delete(ids)
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.post("/reindex")
def reindex(config: dict = Body(...), function: str = Body(default=None)):
    """
    Recreates this embeddings index using config. This method only works if document content storage is enabled.

    Args:
        config: new config
        function: optional function to prepare content for indexing
    """

    try:
        application.get().reindex(config, function)
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.get("/count")
def count():
    """
    Deletes from an embeddings index. Returns list of ids deleted.

    Args:
        ids: list of ids to delete

    Returns:
        ids deleted
    """

    return application.get().count()


@router.post("/explain")
def explain(query: str = Body(...), texts: List[str] = Body(default=None), limit: int = Body(default=None)):
    """
    Explains the importance of each input token in text for a query.

    Args:
        query: query text
        texts: list of text

    Returns:
        list of dict where a higher scores represents higher importance relative to the query
    """

    return application.get().explain(query, texts, limit)


@router.post("/batchexplain")
def batchexplain(queries: List[str] = Body(...), texts: List[str] = Body(default=None), limit: int = Body(default=None)):
    """
    Explains the importance of each input token in text for a query.

    Args:
        query: query text
        texts: list of text

    Returns:
        list of dict where a higher scores represents higher importance relative to the query
    """

    return application.get().batchexplain(queries, texts, limit)


@router.get("/transform")
def transform(text: str):
    """
    Transforms text into an embeddings array.

    Args:
        text: input text

    Returns:
        embeddings array
    """

    return application.get().transform(text)


@router.post("/batchtransform")
def batchtransform(texts: List[str] = Body(...)):
    """
    Transforms list of text into embeddings arrays.

    Args:
        texts: list of text

    Returns:
        embeddings arrays
    """

    return application.get().batchtransform(texts)

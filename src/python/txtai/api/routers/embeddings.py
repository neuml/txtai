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
    Finds documents in the embeddings model most similar to the input query. Returns
    a list of {id: value, score: value} sorted by highest score, where id is the
    document id in the embeddings model.

    Args:
        query: query text
        request: FastAPI request

    Returns:
        list of {id: value, score: value}
    """

    return application.get().search(query, request)


@router.post("/batchsearch")
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

    return application.get().batchsearch(queries, limit)


@router.post("/add")
def add(documents: List[dict] = Body(...)):
    """
    Adds a batch of documents for indexing.

    Args:
        documents: list of {id: value, text: value}
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

"""
Defines API paths for similarity endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.post("/similarity")
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

    return application.get().similarity(query, texts)


@router.post("/batchsimilarity")
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

    return application.get().batchsimilarity(queries, texts)

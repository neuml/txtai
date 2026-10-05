"""
Defines API paths for reranking endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Body

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/rerank")
def rerank(query: str, limit: Optional[int] = 3, factor: Optional[int] = 10):
    """
    Queries an embeddings database and reranks the results with a similarity pipeline.

    Args:
        query: query text
        limit: maximum results
        factor: factor to multiply limit by for the initial embeddings search

    Returns:
        query results
    """

    return application.get().pipeline("reranker", (query, limit, factor))


@router.post("/batchrerank")
def batchrerank(queries: List[str] = Body(...), limit: Optional[int] = Body(default=3), factor: Optional[int] = Body(default=10)):
    """
    Queries an embeddings database and reranks the results with a similarity pipeline.

    Args:
        queries: list of queries
        limit: maximum results
        factor: factor to multiply limit by for the initial embeddings search

    Returns:
        query results
    """

    return application.get().pipeline("reranker", (queries, limit, factor))

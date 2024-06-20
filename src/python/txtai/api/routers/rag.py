"""
Defines API paths for rag endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Body

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/rag")
def rag(query: str, maxlength: Optional[int] = None):
    """
    Runs a RAG pipeline for the input query.

    Args:
        query: input RAG query
        maxlength: optional response max length

    Returns:
        answer
    """

    kwargs = {"maxlength": maxlength} if maxlength else {}
    return application.get().pipeline("rag", query, **kwargs)


@router.post("/batchrag")
def batchrag(queries: List[str] = Body(...), maxlength: Optional[int] = Body(default=None)):
    """
    Runs a RAG pipeline for the input queries.

    Args:
        queries: input RAG queries
        maxlength: optional response max length

    Returns:
        list of answers
    """

    kwargs = {"maxlength": maxlength} if maxlength else {}
    return application.get().pipeline("rag", queries, **kwargs)

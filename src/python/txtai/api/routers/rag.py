"""
Defines API paths for rag endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/rag")
def rag(query: str, maxlength: Optional[int] = None, stream: Optional[bool] = False, stripthink: Optional[bool] = None):
    """
    Runs a RAG pipeline for the input query.

    Args:
        query: input RAG query
        maxlength: optional response max length
        stream: streams response if True
        stripthink: strip thinking tags if True

    Returns:
        answer
    """

    # Build keyword arguments
    params = [("maxlength", maxlength), ("stream", stream), ("stripthink", stripthink)]
    kwargs = {key: value for key, value in params if value}

    # Run pipeline
    result = application.get().pipeline("rag", query, **kwargs)

    # Handle both standard and streaming responses
    return StreamingResponse(result) if stream else result


@router.post("/batchrag")
def batchrag(
    queries: List[str] = Body(...),
    maxlength: Optional[int] = Body(default=None),
    stream: Optional[bool] = Body(default=False),
    stripthink: Optional[bool] = Body(default=None),
):
    """
    Runs a RAG pipeline for the input queries.

    Args:
        queries: input RAG queries
        maxlength: optional response max length
        stream: streams response if True
        stripthink: strip thinking tags if True

    Returns:
        answers
    """

    # Build keyword arguments
    params = [("maxlength", maxlength), ("stream", stream), ("stripthink", stripthink)]
    kwargs = {key: value for key, value in params if value}

    # Run pipeline
    result = application.get().pipeline("rag", queries, **kwargs)

    # Handle both standard and streaming responses
    return StreamingResponse(result) if stream else result

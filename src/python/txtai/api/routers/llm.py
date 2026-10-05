"""
Defines API paths for llm endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/llm")
def llm(text: str, maxlength: Optional[int] = None, stream: Optional[bool] = False, stripthink: Optional[bool] = None):
    """
    Runs a LLM pipeline for the input text.

    Args:
        text: input text
        maxlength: optional response max length
        stream: streams response if True
        stripthink: strip thinking tags if True

    Returns:
        response text
    """

    # Build keyword arguments
    params = [("maxlength", maxlength), ("stream", stream), ("stripthink", stripthink)]
    kwargs = {key: value for key, value in params if value}

    # Run pipeline
    result = application.get().pipeline("llm", text, **kwargs)

    # Handle both standard and streaming responses
    return StreamingResponse(result) if stream else result


@router.post("/batchllm")
def batchllm(
    texts: List[str] = Body(...),
    maxlength: Optional[int] = Body(default=None),
    stream: Optional[bool] = Body(default=False),
    stripthink: Optional[bool] = Body(default=None),
):
    """
    Runs a LLM pipeline for the input texts.

    Args:
        texts: input texts
        maxlength: optional response max length
        stream: streams response if True
        stripthink: strip thinking tags if True

    Returns:
        response texts
    """

    # Build keyword arguments
    params = [("maxlength", maxlength), ("stream", stream), ("stripthink", stripthink)]
    kwargs = {key: value for key, value in params if value}

    # Run pipeline
    result = application.get().pipeline("llm", texts, **kwargs)

    # Handle both standard and streaming responses
    return StreamingResponse(result) if stream else result

"""
Defines API paths for llm endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Body

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/llm")
def llm(text: str, maxlength: Optional[int] = None):
    """
    Runs a LLM pipeline for the input text.

    Args:
        text: input text
        maxlength: optional response max length

    Returns:
        response text
    """

    kwargs = {"maxlength": maxlength} if maxlength else {}
    return application.get().pipeline("llm", text, **kwargs)


@router.post("/batchllm")
def batchllm(texts: List[str] = Body(...), maxlength: Optional[int] = Body(default=None)):
    """
    Runs a LLM pipeline for the input texts.

    Args:
        texts: input texts
        maxlength: optional response max length

    Returns:
        list of response texts
    """

    kwargs = {"maxlength": maxlength} if maxlength else {}
    return application.get().pipeline("llm", texts, **kwargs)

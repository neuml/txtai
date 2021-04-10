"""
Defines API paths for summary endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.get("/summary")
def summary(text: str, minlength: Optional[int] = None, maxlength: Optional[int] = None):
    """
    Runs a summarization model against a block of text.

    Args:
        text: text to summarize
        minlength: minimum length for summary
        maxlength: maximum length for summary

    Returns:
        summary text
    """

    return application.get().pipeline("summary", (text, minlength, maxlength))


@router.post("/batchsummary")
def batchsummary(texts: List[str] = Body(...), minlength: Optional[int] = Body(default=None), maxlength: Optional[int] = Body(default=None)):
    """
    Runs a summarization model against a block of text.

    Args:
        texts: list of text to summarize
        minlength: minimum length for summary
        maxlength: maximum length for summary

    Returns:
        list of summary text
    """

    return application.get().pipeline("summary", (texts, minlength, maxlength))

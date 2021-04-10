"""
Defines API paths for extractor endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.post("/extract")
def extract(queue: List[dict] = Body(...), texts: List[str] = Body(...)):
    """
    Extracts answers to input questions.

    Args:
        queue: list of {name: value, query: value, question: value, snippet: value}
        texts: list of text

    Returns:
        list of {name: value, answer: value}
    """

    return application.get().extract(queue, texts)

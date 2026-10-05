"""
Defines API paths for extractor endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Body

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.post("/extract")
def extract(queue: List[dict] = Body(...), texts: Optional[List[str]] = Body(default=None)):
    """
    Extracts answers to input questions.

    Args:
        queue: list of {name: value, query: value, question: value, snippet: value}
        texts: optional list of text

    Returns:
        list of {name: value, answer: value}
    """

    return application.get().extract(queue, texts)

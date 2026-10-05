"""
Defines API paths for entity endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/entity")
def entity(text: str):
    """
    Applies a token classifier to text.

    Args:
        text: input text

    Returns:
        list of (entity, entity type, score) per text element
    """

    return application.get().pipeline("entity", (text,))


@router.post("/batchentity")
def batchentity(texts: List[str] = Body(...)):
    """
    Applies a token classifier to text.

    Args:
        texts: list of text

    Returns:
        list of (entity, entity type, score) per text element
    """

    return application.get().pipeline("entity", (texts,))

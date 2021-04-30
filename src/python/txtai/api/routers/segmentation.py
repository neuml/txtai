"""
Defines API paths for segmentation endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.get("/segment")
def segment(text: str):
    """
    Segments text into semantic units.

    Args:
        text: input text

    Returns:
        segmented text
    """

    return application.get().pipeline("segmentation", (text,))


@router.post("/batchsegment")
def batchsegment(texts: List[str] = Body(...)):
    """
    Segments text into semantic units.

    Args:
        texts: list of texts to segment

    Returns:
        list of segmented text
    """

    return application.get().pipeline("segmentation", (texts,))

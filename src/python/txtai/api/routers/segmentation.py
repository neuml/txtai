"""
Defines API paths for segmentation endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.get("/segment")
def textract(text: str):
    """
    Segments input text.

    Args:
        text: input text

    Returns:
        segmented text
    """

    return application.get().pipeline("segmentation", (text,))


@router.post("/batchsegment")
def batchsegment(texts: List[str] = Body(...)):
    """
    Extracts text from a file at path.

    Args:
        text: list of texts to segment

    Returns:
        list of segmented text
    """

    return application.get().pipeline("segmentation", (texts,))

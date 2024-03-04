"""
Defines API paths for caption endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/caption")
def caption(file: str):
    """
    Builds captions for images.

    Args:
        file: file to process

    Returns:
        list of captions
    """

    return application.get().pipeline("caption", (file,))


@router.post("/batchcaption")
def batchcaption(files: List[str] = Body(...)):
    """
    Builds captions for images.

    Args:
        files: list of files to process

    Returns:
        list of captions
    """

    return application.get().pipeline("caption", (files,))

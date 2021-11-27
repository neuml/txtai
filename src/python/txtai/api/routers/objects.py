"""
Defines API paths for objects endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.get("/objects")
def objects(file: str):
    """
    Applies object detection/image classification models to images.

    Args:
        file: file to process

    Returns:
        list of (label, score) elements
    """

    return application.get().pipeline("objects", (file,))


@router.post("/batchobjects")
def batchobjects(files: List[str] = Body(...)):
    """
    Applies object detection/image classification models to images.

    Args:
        files: list of files to process

    Returns:
        list of (label, score) elements
    """

    return application.get().pipeline("objects", (files,))

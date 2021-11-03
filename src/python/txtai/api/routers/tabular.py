"""
Defines API paths for tabular endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.get("/tabular")
def tabular(file: str):
    """
    Splits tabular data into rows and columns.

    Args:
        file: file to process

    Returns:
        list of (id, text, tag) elements
    """

    return application.get().pipeline("tabular", (file,))


@router.post("/batchtabular")
def batchtabular(files: List[str] = Body(...)):
    """
    Splits tabular data into rows and columns.

    Args:
        files: list of files to process

    Returns:
        list of (id, text, tag) elements
    """

    return application.get().pipeline("tabular", (files,))

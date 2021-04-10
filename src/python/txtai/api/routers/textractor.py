"""
Defines API paths for textractor endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.get("/textract")
def textract(file: str):
    """
    Extracts text from a file at path.

    Args:
        file: file to extract text

    Returns:
        extracted text
    """

    return application.get().pipeline("textractor", (file,))


@router.post("/batchtextract")
def batchtextract(files: List[str] = Body(...)):
    """
    Extracts text from a file at path.

    Args:
        files: list of files to extract text

    Returns:
        list of extracted text
    """

    return application.get().pipeline("textractor", (files,))

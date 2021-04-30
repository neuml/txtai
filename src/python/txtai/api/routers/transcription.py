"""
Defines API paths for transcription endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.get("/transcribe")
def transcribe(file: str):
    """
    Transcribes audio files to text.

    Args:
        file: file to transcribe

    Returns:
        transcribed text
    """

    return application.get().pipeline("transcription", (file,))


@router.post("/batchtranscribe")
def batchtranscribe(files: List[str] = Body(...)):
    """
    Transcribes audio files to text.

    Args:
        files: list of files to transcribe

    Returns:
        list of transcribed text
    """

    return application.get().pipeline("transcription", (files,))

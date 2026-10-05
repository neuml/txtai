"""
Defines API paths for TTS endpoints
"""

from typing import Optional

from fastapi import APIRouter, Response

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/texttospeech")
def texttospeech(text: str, speaker: Optional[str] = None, encoding: Optional[str] = "mp3"):
    """
    Generates speech from text.

    Args:
        text: text
        speaker: speaker id, defaults to 1
        encoding: optional audio encoding format

    Returns:
        Audio data
    """

    # Convert to audio
    audio = application.get().pipeline("texttospeech", text, speaker=speaker, encoding=encoding)

    # Write audio
    return Response(audio, headers={"Content-Disposition": f"attachment;filename=speech.{encoding.lower()}"})

"""
Defines API paths for translation endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.get("/translate")
def translate(text: str, target: Optional[str] = "en", source: Optional[int] = None):
    """
    Translates text from source language into target language.

    Args:
        text: text to translate
        target: target language code, defaults to "en"
        source: source language code, detects language if not provided

    Returns:
        translated text
    """

    return application.get().pipeline("translation", (text, target, source))


@router.post("/batchtranslate")
def batchtranslate(texts: List[str] = Body(...), target: Optional[str] = Body(default="en"), source: Optional[str] = Body(default=None)):
    """
    Translates text from source language into target language.

    Args:
        texts: list of text to translate
        target: target language code, defaults to "en"
        source: source language code, detects language if not provided

    Returns:
        list of translated text
    """

    return application.get().pipeline("translation", (texts, target, source))

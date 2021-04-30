"""
Defines API paths for labels endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application

router = APIRouter()


@router.post("/label")
def label(text: str = Body(...), labels: List[str] = Body(...)):
    """
    Applies a zero shot classifier to text using a list of labels. Returns a list of
    {id: value, score: value} sorted by highest score, where id is the index in labels.

    Args:
        text: input text
        labels: list of labels

    Returns:
        list of {id: value, score: value} per text element
    """

    return application.get().label(text, labels)


@router.post("/batchlabel")
def batchlabel(texts: List[str] = Body(...), labels: List[str] = Body(...)):
    """
    Applies a zero shot classifier to list of text using a list of labels. Returns a list of
    {id: value, score: value} sorted by highest score, where id is the index in labels per
    text element.

    Args:
        texts: list of text
        labels: list of labels

    Returns:
        list of {id: value score: value} per text element
    """

    return application.get().label(texts, labels)

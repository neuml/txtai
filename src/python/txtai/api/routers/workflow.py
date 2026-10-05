"""
Defines API paths for workflow endpoints.
"""

from typing import List

from fastapi import APIRouter, Body

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.post("/workflow")
def workflow(name: str = Body(...), elements: List = Body(...)):
    """
    Executes a named workflow using elements as input.

    Args:
        name: workflow name
        elements: list of elements to run through workflow

    Returns:
        list of processed elements
    """

    return application.get().workflow(name, elements)

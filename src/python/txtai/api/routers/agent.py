"""
Defines API paths for agent endpoints.
"""

from typing import Optional

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


@router.post("/agent")
def agent(name: str = Body(...), text: str = Body(...), maxlength: Optional[int] = Body(default=None), stream: Optional[bool] = Body(default=None)):
    """
    Executes a named agent for input text.

    Args:
        name: agent name
        text: instructions to run
        maxlength: maximum sequence length
        stream: stream response if True, defaults to False

    Returns:
        response text
    """

    # Build keyword arguments
    kwargs = {key: value for key, value in [("stream", stream), ("maxlength", maxlength)] if value}

    # Run agent
    result = application.get().agent(name, text, **kwargs)

    # Handle both standard and streaming responses
    return StreamingResponse(result) if stream else result

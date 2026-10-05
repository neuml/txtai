"""
Route module
"""

import json

from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

from .responses import ResponseFactory, MessagePackResponse


class EncodingAPIRoute(APIRoute):
    """
    Extended APIRoute that encodes responses based on HTTP Accept header.
    """

    def get_route_handler(self):
        """
        Resolves a response class based on the HTTP Accept header.

        Returns:
            route handler function
        """

        # Get handle to the original route handler
        original = super().get_route_handler()

        async def handler(request):
            # Target response class
            target = ResponseFactory.create(request)
            request.state.response_class = target

            # Get response
            response = await original(request)

            # Force MessagePackResponse when it's requested and response type doesn't match
            return (
                target(
                    content=json.loads(response.body),
                    status_code=response.status_code,
                    headers={k: v for k, v in response.headers.items() if k.lower() not in ["content-length", "content-type"]},
                )
                if not isinstance(response, (target, StreamingResponse)) and target == MessagePackResponse
                else response
            )

        return handler

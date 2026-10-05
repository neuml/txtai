"""
Factory module
"""

from .json import JSONResponse
from .messagepack import MessagePackResponse


class ResponseFactory:
    """
    Methods to create Response classes.
    """

    @staticmethod
    def create(request):
        """
        Gets a response class for request using the Accept header.

        Args:
            request: request

        Returns:
            response class
        """

        # Get Accept header
        accept = request.headers.get("Accept")

        # Get response class
        return MessagePackResponse if accept == MessagePackResponse.media_type else JSONResponse

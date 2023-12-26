"""
Authorization module
"""

import hashlib
import os

from fastapi import Header, HTTPException


class Authorization:
    """
    Basic token authorization.
    """

    def __init__(self, token=None):
        """
        Creates a new Authorization instance.

        Args:
            token: SHA-256 hash of token to check
        """

        self.token = token if token else os.environ.get("TOKEN")

    def __call__(self, authorization: str = Header(default=None)):
        """
        Validates authorization header is present and equal to current token.

        Args:
            authorization: authorization header
        """

        if not authorization or self.token != self.digest(authorization):
            raise HTTPException(status_code=401, detail="Invalid Authorization Token")

    def digest(self, authorization):
        """
        Computes a SHA-256 hash for input authorization token.

        Args:
            authorization: authorization header

        Returns:
            SHA-256 hash of authorization token
        """

        # Replace Bearer prefix
        prefix = "Bearer "
        token = authorization[len(prefix) :] if authorization.startswith(prefix) else authorization

        # Compute SHA-256 hash
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

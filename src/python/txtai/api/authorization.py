"""
Authorization module
"""

import hashlib
import hmac

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class Authorization:
    """
    Basic token authorization.
    """

    def __init__(self, token):
        """
        Creates a new Authorization instance.

        Args:
            token: SHA-256 hash of the valid authorization token
        """

        self.token = token

    def __call__(self, authorization: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """
        Validates authorization header is present and equal to current token.

        Args:
            authorization: authorization header
        """

        # Authorization header is parsed into {scheme="Bearer", credentials="<token>"}
        if not hmac.compare_digest(self.token, self.digest(authorization.credentials)):
            raise HTTPException(status_code=401, detail="Invalid Authorization Token")

    def digest(self, token):
        """
        Computes a SHA-256 hash for input authorization token.

        Args:
            token: authorization token

        Returns:
            SHA-256 hash of authorization token
        """

        return hashlib.sha256(token.encode("utf-8")).hexdigest()

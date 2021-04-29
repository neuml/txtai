"""
ServiceTask module
"""

import requests

from .base import Task


class ServiceTask(Task):
    """
    Task to runs requests against remote service urls.
    """

    def __init__(self, url=None, method=None, params=None, select=None, unpack=True):
        super().__init__(lambda x: self.request(url, method, params, x), select, unpack)

    def request(self, url, method, params, data):
        """
        Execute service request.

        Args:
            url: service url
            method: method (get or post)
            params: dict of constant parameters to pass to request
            data: dynamic data for this specific request

        Returns:
            response as JSON
        """

        if not params:
            params = data
        else:
            # Create copy of parameters
            params = params.copy()

            # Add data to parameters
            for key in params:
                if not params[key]:
                    params[key] = data

        if method == "get":
            return requests.get(url, params=params).json()

        return requests.post(url, json=params).json()

"""
ServiceTask module
"""

# Conditional import
try:
    import requests
    import xmltodict

    XML_TO_DICT = True
except ImportError:
    XML_TO_DICT = False

from .base import Task


class ServiceTask(Task):
    """
    Task to runs requests against remote service urls.
    """

    def register(self, url=None, method=None, params=None, batch=True, extract=None):
        """
        Adds service parameters to task. Checks if required dependencies are installed.

        Args:
            url: url to connect to
            method: http method, GET or POST
            params: default query parameters
            batch: if True, all elements are passed in a single batch request, otherwise a service call is executed per element
            extract: list of sections to extract from response
        """

        if not XML_TO_DICT:
            raise ImportError('ServiceTask is not available - install "workflow" extra to enable')

        # pylint: disable=W0201
        # Save URL, method and parameter defaults
        self.url = url
        self.method = method
        self.params = params

        # If True, all elements are passed in a single batch request, otherwise a service call is executed per element
        self.batch = batch

        # Save sections to extract. Supports both a single string and a hierarchical list of sections.
        self.extract = extract
        if self.extract:
            self.extract = [self.extract] if isinstance(self.extract, str) else self.extract

    def execute(self, elements, executor=None):
        if self.batch:
            elements = self.request(elements)
        else:
            elements = [self.request(element) for element in elements]

        return super().execute(elements, executor)

    def request(self, data):
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

        if not self.params:
            params = data
        else:
            # Create copy of parameters
            params = self.params.copy()

            # Add data to parameters
            for key in params:
                if not params[key]:
                    params[key] = data

        # Run request
        if self.method and self.method.lower() == "get":
            response = requests.get(self.url, params=params)
        else:
            response = requests.post(self.url, json=params)

        # Parse data based on content-type
        mimetype = response.headers["Content-Type"].split(";")[0]
        if mimetype.lower().endswith("xml"):
            data = xmltodict.parse(response.text)
        else:
            data = response.json()

        # Extract content from response, if necessary
        if self.extract:
            for tag in self.extract:
                data = data[tag]

        return data

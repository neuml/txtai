"""
OpenCode module
"""

import httpx

from transformers.utils import cached_file

from .generation import Generation


class OpenCode(Generation):
    """
    OpenCode generative model.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is an OpenCode model.

        Args:
            path: input path

        Returns:
            True if this is an OpenCode model, False otherwise
        """

        return isinstance(path, str) and path.lower().startswith("opencode") and not OpenCode.ishub(path)

    @staticmethod
    def ishub(path):
        """
        Checks if path is available on the HF Hub.

        Args:
            input path

        Returns:
            True if this is a model on the HF Hub
        """

        # pylint: disable=W0702
        try:
            return cached_file(path_or_repo_id=path, filename="config.json") is not None if "/" in path else False
        except:
            return False

    def __init__(self, path, template=None, **kwargs):
        super().__init__(path, template, **kwargs)

        # Get model and provider from path
        self.provider, self.model = path.split("/", 1) if "/" in path else (None, None)

        # Get base url, default to `opencode serve` default
        self.url = kwargs.get("url", "http://localhost:4096")

        # Start an OpenCode session
        self.session = httpx.post(f"{self.url}/session").json()

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        for text in texts:
            # Build text string
            text = "\n".join([x["content"] for x in text]) if isinstance(text, list) else text

            # Add text to request
            request = {"parts": [{"type": "text", "text": text}]}

            # Optionally add model, if available
            if self.model:
                request["model"] = {"providerID": self.provider, "modelID": self.model}

            # Submit request and read JSON response
            response = httpx.post(f"{self.url}/session/{self.session['id']}/message", json=request, timeout=None).json()

            # Transform JSON into a text response
            yield "\n".join([part["text"] for part in response["parts"] if part["type"] == "text"])

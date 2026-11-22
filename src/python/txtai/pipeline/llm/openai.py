"""
OpenAI module
"""

# Conditional import
try:
    import openai as api

    OPENAI = True
except ImportError:
    OPENAI = False

from .generation import Generation

from ...util import Download


class OpenAI(Generation):
    """
    OpenAI API generative model. Supports the OpenAI API and OpenAI-compatible API endpoints.
    """

    # OpenAI client parameters, all other parameters are passed to the generation call
    CLIENT = ("api_key", "base_url", "organization", "project", "timeout", "max_retries", "default_headers", "default_query", "http_client")

    @staticmethod
    def ismodel(path):
        """
        Checks if path is an OpenAI model.

        Args:
            path: input path

        Returns:
            True if this is an OpenAI model, False otherwise
        """

        return isinstance(path, str) and OPENAI and path.lower().startswith("openai") and not OpenAI.ishub(path)

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
            return Download()(path, "config.json") is not None if "/" in path else False
        except:
            return False

    def __init__(self, path, template=None, **kwargs):
        super().__init__(path, template, **kwargs)

        if not OPENAI:
            raise ImportError('OpenAI is not available - install "pipeline" extra to enable')

        # Ignore common pipeline parameters. Accept api_base as an alias for base_url.
        kwargs = {k: v for k, v in self.kwargs.items() if k not in ["quantize", "gpu", "model", "task"]}
        kwargs = {("base_url" if k == "api_base" else k): v for k, v in kwargs.items()}

        # Split client and generation parameters
        self.client = api.OpenAI(**{k: v for k, v in kwargs.items() if k in OpenAI.CLIENT})
        self.kwargs = {k: v for k, v in kwargs.items() if k not in OpenAI.CLIENT}

        # Remove openai provider prefix from model name
        self.model = path[len("openai/") :] if path.lower().startswith("openai/") else path

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        for text in texts:
            # Chat messages use the chat completions API, raw prompts use the completions API
            method = self.client.chat.completions.create if isinstance(text, list) else self.client.completions.create
            inputs = {"messages": text} if isinstance(text, list) else {"prompt": text}

            # LLM API call
            result = method(model=self.model, **self.parameters(inputs, maxlength, stream, stop, **kwargs))

            # Stream response. Skip chunks without choices (i.e. usage and content filter chunks).
            yield from self.response(x.model_dump() for x in (result if stream else [result]) if x.choices)

    def parameters(self, inputs, maxlength, stream, stop, **kwargs):
        """
        Builds API call parameters.

        Args:
            inputs: messages or prompt parameter
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings
            kwargs: additional generation keyword arguments

        Returns:
            API call parameters
        """

        parameters = {"max_tokens": maxlength, "stream": stream, "stop": stop, **inputs, **self.kwargs, **kwargs}

        # Reasoning models reject max_tokens when max_completion_tokens is set
        if "max_completion_tokens" in parameters:
            parameters.pop("max_tokens")

        # Drop unset parameters
        return {k: v for k, v in parameters.items() if v is not None}

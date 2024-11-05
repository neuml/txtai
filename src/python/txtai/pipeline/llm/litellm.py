"""
LiteLLM module
"""

# Conditional import
try:
    import litellm as api

    LITELLM = True
except ImportError:
    LITELLM = False

from .generation import Generation


class LiteLLM(Generation):
    """
    LiteLLM generative model.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is a LiteLLM model.

        Args:
            path: input path

        Returns:
            True if this is a LiteLLM model, False otherwise
        """

        # pylint: disable=W0702
        if isinstance(path, str) and LITELLM:
            debug = api.suppress_debug_info
            try:
                # Suppress debug messages for this test
                api.suppress_debug_info = True
                return api.get_llm_provider(path)
            except:
                return False
            finally:
                # Restore debug info value to original value
                api.suppress_debug_info = debug

        return False

    def __init__(self, path, template=None, **kwargs):
        super().__init__(path, template, **kwargs)

        if not LITELLM:
            raise ImportError('LiteLLM is not available - install "pipeline" extra to enable')

        # Ignore common pipeline parameters
        self.kwargs = {k: v for k, v in self.kwargs.items() if k not in ["quantize", "gpu", "model", "task"]}

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        for text in texts:
            # LLM API call
            result = api.completion(
                model=self.path,
                messages=[{"content": text, "role": "prompt"}] if isinstance(text, str) else text,
                max_tokens=maxlength,
                stream=stream,
                stop=stop,
                **{**self.kwargs, **kwargs}
            )

            # Stream response
            yield from self.response(result if stream else [result])

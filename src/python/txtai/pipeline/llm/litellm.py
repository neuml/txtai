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

    def execute(self, texts, maxlength, **kwargs):
        results = []
        for text in texts:
            result = api.completion(
                model=self.path,
                messages=[{"content": text, "role": "prompt"}] if isinstance(text, str) else text,
                max_tokens=maxlength,
                **{**kwargs, **self.kwargs}
            )
            results.append(result["choices"][0]["message"]["content"])

        return results

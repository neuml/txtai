"""
Generator module
"""

from .llm import LLM


class Generator(LLM):
    """
    Generate text with a causal language model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        super().__init__(path, quantize, gpu, model, "language-generation", **kwargs)

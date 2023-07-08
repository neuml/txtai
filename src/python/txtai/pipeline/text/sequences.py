"""
Sequences module
"""

from .llm import LLM


class Sequences(LLM):
    """
    Runs text through a sequence-sequence model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        super().__init__(path, quantize, gpu, model, "sequence-sequence", **kwargs)

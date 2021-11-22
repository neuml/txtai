"""
No-Op module
"""

from .base import Pipeline


class Nop(Pipeline):
    """
    Simple no-op pipeline that returns inputs
    """

    def __call__(self, inputs):
        return inputs

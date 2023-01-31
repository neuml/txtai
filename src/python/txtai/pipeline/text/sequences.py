"""
Sequences module
"""

from .generator import Generator


class Sequences(Generator):
    """
    Runs text through a sequence-sequence model.
    """

    def task(self):
        return "text2text-generation"

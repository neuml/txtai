"""
Tokenizer module
"""

import re
import string

from ..base import Pipeline


class Tokenizer(Pipeline):
    """
    Tokenizes text into a list of tokens. Primarily designed for English text.
    """

    # fmt: off
    # English Stop Word List (Standard stop words used by Apache Lucene)
    STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
                  "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
                  "they", "this", "to", "was", "will", "with"}
    # fmt: on

    @staticmethod
    def tokenize(text):
        """
        Tokenizes input text into a list of tokens. Filters tokens that match a specific pattern and removes stop words.

        Args:
            text: input text

        Returns:
            list of tokens
        """

        return Tokenizer()(text)

    def __call__(self, text):
        """
        Tokenizes input text into a list of tokens. Filters tokens that match a specific pattern and removes stop words.

        Args:
            text: input text

        Returns:
            list of tokens
        """

        # Convert to all lowercase, split on whitespace, strip punctuation
        tokens = [token.strip(string.punctuation) for token in text.lower().split()]

        # Tokenize on alphanumeric strings.
        # Require strings to be at least 2 characters long.
        # Require at least 1 non-trailing alpha character in string.
        return [token for token in tokens if re.match(r"^\d*[a-z][\-.0-9:_a-z]{1,}$", token) and token not in Tokenizer.STOP_WORDS]

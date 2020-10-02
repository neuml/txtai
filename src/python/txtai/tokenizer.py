"""
Text tokenization methods
"""

import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


class Tokenizer(object):
    """
    Text tokenization methods
    """

    # English Stop Word List (Standard stop words by NLTK Corpus)
    STOP_WORDS = stopwords.words('english')

    @staticmethod
    def tokenize(text):
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
        # Require at least 1 alpha character in string.
        return [token for token in tokens if re.match(r"^\d*[a-z][\-.0-9:_a-z]{1,}$", token) and token not in Tokenizer.STOP_WORDS]

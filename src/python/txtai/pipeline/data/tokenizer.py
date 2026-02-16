"""
Tokenizer module
"""

import re
import string

import regex

from ..base import Pipeline


class Tokenizer(Pipeline):
    """
    Tokenizes text into tokens using one of the following methods.

      1. Split using word boundary rules from the Unicode Text Segmentation algorithm (see Unicode Standard Annex #29).
         This is similar to the standard tokenizer in Apache Lucene and works well for most languages.

      2. Tokenization method that only accepts alphanumeric tokens from the Latin alphabet. This is a backwards compatible mode
         that was the default with older versions of txtai.

      3. Tokenize on whitespace

      4. Tokenize using a provided regular expression
    """

    # fmt: off
    # English Stop Word List (Standard stop words used by Apache Lucene)
    STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
                  "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
                  "they", "this", "to", "was", "will", "with"}
    # fmt: on

    @staticmethod
    def tokenize(text, lowercase=True, emoji=True, alphanum=True, stopwords=True, whitespace=False, regexp=None, ngrams=None):
        """
        Tokenizes text into a list of tokens. The default backwards compatible parameters filter out English stop words and only
        accept alphanumeric tokens.

        Args:
            text: input text
            lowercase: lower cases all tokens if True, defaults to True
            emoji: tokenize emoji in text if True, defaults to True
            alphanum: requires 2+ character alphanumeric tokens if True, defaults to True
            stopwords: removes provided stop words if a list, removes default English stop words if True, defaults to True
            whitespace: tokenize on whitespace if True, defaults to False
            regexp: tokenize using the provided regular expression, defaults to None
            ngrams: tokenize into ngrams, defaults to None, supports int or dict

        Returns:
            list of tokens
        """

        # Create a tokenizer with backwards compatible settings
        return Tokenizer(lowercase, emoji, alphanum, stopwords, whitespace, regexp, ngrams)(text)

    def __init__(self, lowercase=True, emoji=True, alphanum=False, stopwords=False, whitespace=False, regexp=None, ngrams=None):
        """
        Creates a new tokenizer. The default parameters segment text per Unicode Standard Annex #29.

        Args:
            lowercase: lower cases all tokens if True, defaults to True
            emoji: tokenize emoji in text if True, defaults to True
            alphanum: requires 2+ character alphanumeric tokens if True, defaults to False
            stopwords: removes provided stop words if a list, removes default English stop words if True, defaults to False
            whitespace: tokenize on whitespace if True, defaults to False
            regexp: tokenize using the provided regular expression, defaults to None
            ngrams: tokenize into ngrams, defaults to None, supports int or dict
        """

        # Lowercase
        self.lowercase = lowercase

        # Text segmentation
        self.alphanum, self.whitespace, self.regexp, self.ngrams, self.segment = None, whitespace, None, None, None
        if alphanum:
            # Alphanumeric regex that accepts tokens that meet following rules:
            #  - Strings to be at least 2 characters long AND
            #  - At least 1 non-trailing alpha character in string
            # Note: The standard Python re module is much faster than regex for this expression
            self.alphanum = re.compile(r"^\d*[a-z][\-.0-9:_a-z]{1,}$")
        elif regexp:
            # Regular expression for tokenization
            self.regexp = regex.compile(regexp)
        elif ngrams:
            # Ngram tokenization configuration
            self.ngrams = ngrams if isinstance(ngrams, dict) else {"ngrams": ngrams}
        else:
            # Text segmentation per Unicode Standard Annex #29
            pattern = r"\w\p{Extended_Pictographic}\p{WB:RegionalIndicator}" if emoji else r"\w"
            self.segment = regex.compile(rf"[{pattern}](?:\B\S)*", flags=regex.WORD)

        # Stop words
        self.stopwords = stopwords if isinstance(stopwords, list) else Tokenizer.STOP_WORDS if stopwords else False

    def __call__(self, text):
        """
        Tokenizes text into a list of tokens.

        Args:
            text: input text

        Returns:
            list of tokens
        """

        # Check for None and skip processing
        if text is None:
            return None

        # Lowercase
        text = text.lower() if self.lowercase else text

        if self.alphanum:
            # Text segmentation using standard split
            tokens = [token.strip(string.punctuation) for token in text.split()]

            # Filter on alphanumeric strings.
            tokens = [token for token in tokens if re.match(self.alphanum, token)]
        elif self.whitespace:
            # Text segmentation using whitespace
            tokens = text.split()
        elif self.regexp:
            # Text segmentation using a custom regular expression
            tokens = regex.findall(self.regexp, text)
        elif self.ngrams:
            # Ngram tokenizer
            tokens = self.ngramtokenize(text)
        else:
            # Text segmentation per Unicode Standard Annex #29
            tokens = regex.findall(self.segment, text)

        # Stop words
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        return tokens

    def ngramtokenize(self, text):
        """
        Tokenizes input text into ngrams.

        Args:
            text: input text

        Return:
            [ngrams]
        """

        # Ngram configuration
        number = self.ngrams.get("ngrams", 3)
        lpad = self.ngrams.get("lpad", "")
        rpad = self.ngrams.get("rpad", "")
        unique = self.ngrams.get("unique", False)

        # Split on non-whitespace and apply optional word padding
        words = [f"{lpad}{x}{rpad}" for x in re.split(r"\W+", text.lower()) if x.strip()]

        # Generate ngrams
        ngrams = []
        for word in words:
            for x in range(0, len(word) - number + 1):
                ngrams.append(word[x : x + number])

        # Reduce to unique ngrams, if necessary and return
        return list(set(ngrams)) if unique else ngrams

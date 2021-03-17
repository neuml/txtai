"""
Textractor module
"""

import re

from nltk import sent_tokenize
from tika import parser

from .base import Pipeline


class Textractor(Pipeline):
    """
    Extracts text from files.
    """

    def __init__(self, sentences=False, paragraphs=False, minlength=None, join=False):
        """
        Creates a new Textractor.

        Args:
            sentences: tokenize text into sentences if True, defaults to False
            paragraphs: tokenizes text into paragraphs if True, defaults to False
            minlength: require at least minlength characters per text element, defaults to None
            join: joins tokenized sections back together if True, defaults to False
        """

        self.sentences = sentences
        self.paragraphs = paragraphs
        self.minlength = minlength
        self.join = join

    def __call__(self, files):
        """
        Extracts text from a file at path.

        Args:
            files: text|list

        Returns:
            extracted text from files
        """

        files = [files] if not isinstance(files, list) else files

        content = []
        for path in files:
            parsed = parser.from_file(path)
            text = parsed["content"]

            if text:
                # Parse and add extracted results
                content.append(self.parse(text))

        return content

    def parse(self, text):
        """
        Splits and cleans text based on the current parameters.

        Args:
            text: input text

        Returns:
            parsed and clean content
        """

        content = None

        if self.sentences:
            content = [self.clean(x) for x in sent_tokenize(text)]
        elif self.paragraphs:
            content = [self.clean(x) for x in text.split("\n\n")]
        else:
            content = [self.clean(text)]

        # Remove empty strings
        content = [x for x in content if x]

        if self.sentences or self.paragraphs:
            return " ".join(content) if self.join else content

        return content[0]

    def clean(self, text):
        """
        Applies a series of rules to clean extracted text.

        Args:
            text: input text

        Returns:
            clean text
        """

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text if not self.minlength or len(text) >= self.minlength else None

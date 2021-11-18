"""
Segmentation module
"""

import re

# Conditional import
try:
    from nltk import sent_tokenize

    NLTK = True
except ImportError:
    NLTK = False

from ..base import Pipeline


class Segmentation(Pipeline):
    """
    Segments text into logical units.
    """

    def __init__(self, sentences=False, lines=False, paragraphs=False, minlength=None, join=False):
        """
        Creates a new Segmentation pipeline.

        Args:
            sentences: tokenize text into sentences if True, defaults to False
            lines: tokenizes text into lines if True, defaults to False
            paragraphs: tokenizes text into paragraphs if True, defaults to False
            minlength: require at least minlength characters per text element, defaults to None
            join: joins tokenized sections back together if True, defaults to False
        """

        if not NLTK:
            raise ImportError('Segmentation pipeline is not available - install "pipeline" extra to enable')

        self.sentences = sentences
        self.lines = lines
        self.paragraphs = paragraphs
        self.minlength = minlength
        self.join = join

    def __call__(self, text):
        """
        Segments text into semantic units.

        This method supports text as a string or a list. If the input is a string, the return
        type is text|list. If text is a list, a list of returned, this could be a
        list of text or a list of lists depending on the tokenization strategy.

        Args:
            text: text|list

        Returns:
            segmented text
        """

        # Get inputs
        texts = [text] if not isinstance(text, list) else text

        # Extract text for each input file
        results = []
        for value in texts:
            # Get text
            value = self.text(value)

            # Parse and add extracted results
            results.append(self.parse(value))

        return results[0] if isinstance(text, str) else results

    def text(self, text):
        """
        Hook to allow extracting text out of input text object.

        Args:
            text: object to extract text from
        """

        return text

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
        elif self.lines:
            content = [self.clean(x) for x in text.split("\n")]
        elif self.paragraphs:
            content = [self.clean(x) for x in text.split("\n\n")]
        else:
            content = [self.clean(text)]

        # Remove empty strings
        content = [x for x in content if x]

        if self.sentences or self.lines or self.paragraphs:
            return " ".join(content) if self.join else content

        return content[0] if content else content

    def clean(self, text):
        """
        Applies a series of rules to clean text.

        Args:
            text: input text

        Returns:
            clean text
        """

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text if not self.minlength or len(text) >= self.minlength else None

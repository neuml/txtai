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

    def __init__(self, sentences=False, lines=False, paragraphs=False, minlength=None, join=False, sections=False):
        """
        Creates a new Segmentation pipeline.

        Args:
            sentences: tokenize text into sentences if True, defaults to False
            lines: tokenizes text into lines if True, defaults to False
            paragraphs: tokenizes text into paragraphs if True, defaults to False
            minlength: require at least minlength characters per text element, defaults to None
            join: joins tokenized sections back together if True, defaults to False
            sections: tokenizes text into sections if True, defaults to False. Splits using section or page breaks, depending on what's available
        """

        if not NLTK:
            raise ImportError('Segmentation pipeline is not available - install "pipeline" extra to enable')

        self.sentences = sentences
        self.lines = lines
        self.paragraphs = paragraphs
        self.sections = sections
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
            content = [self.clean(x) for x in re.split(r"\n{1,}", text)]
        elif self.paragraphs:
            content = [self.clean(x) for x in re.split(r"\n{2,}", text)]
        elif self.sections:
            split = r"\f" if "\f" in text else r"\n{3,}"
            content = [self.clean(x) for x in re.split(split, text)]
        else:
            content = self.clean(text)

        # Text tokenization enabled
        if isinstance(content, list):
            # Remove empty strings
            content = [x for x in content if x]
            return " ".join(content) if self.join else content

        # Default method that returns clean text
        return content

    def clean(self, text):
        """
        Applies a series of rules to clean text.

        Args:
            text: input text

        Returns:
            clean text
        """

        text = re.sub(r" +", " ", text)
        text = text.strip()

        return text if not self.minlength or len(text) >= self.minlength else None

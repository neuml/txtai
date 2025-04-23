"""
Textractor module tests
"""

import platform
import unittest

from txtai.pipeline import Textractor

# pylint: disable = C0411
from utils import Utils


class TestTextractor(unittest.TestCase):
    """
    Textractor tests.
    """

    def testClean(self):
        """
        Test text cleaning method
        """

        # Default text cleaning
        textractor = Textractor()
        self.assertEqual(textractor(" a  b  c "), "a b c")

        # Require text to be minlength
        textractor = Textractor(minlength=10)
        self.assertEqual(textractor(" a  b  c "), None)

        # Disable text cleaning
        textractor = Textractor(cleantext=False, minlength=10)
        self.assertEqual(textractor(" a  b  c "), " a  b  c ")

    def testChonkie(self):
        """
        Test a chonkie chunker
        """

        # Test chonkie chunking
        textractor = Textractor(chunker="sentence", chunk_size=5, chunk_overlap=0)
        self.assertEqual(textractor("This is a test. And another test."), ["This is a test.", "And another test."])

        # Test bad chunker throws an exception
        with self.assertRaises(AttributeError):
            textractor = Textractor(chunker="badchunker")

    def testDefault(self):
        """
        Test default text extraction
        """

        # Text input
        textractor = Textractor(backend=None)
        text = textractor(Utils.PATH + "/tabular.csv")
        self.assertEqual(len(text), 125)

        # Markdown input
        textractor = Textractor(sections=True)
        sections = textractor("# Heading 1\nText1\n\n# Heading 2\nText2\n")

        # Check number of sections is as expected
        self.assertEqual(len(sections), 2)

    @unittest.skipIf(platform.system() == "Darwin", "Docling skipped on macOS to avoid MPS issues")
    def testDocling(self):
        """
        Test docling backend
        """

        textractor = Textractor(backend="docling")

        # Extract text and check for Markdown formatting
        text = textractor(Utils.PATH + "/article.pdf")
        self.assertTrue("## Introducing txtai" in text)

    def testLines(self):
        """
        Test extraction to lines
        """

        textractor = Textractor(lines=True)

        # Extract text as lines
        lines = textractor(Utils.PATH + "/article.pdf")

        # Check number of lines is as expected
        self.assertEqual(len(lines), 35)

    def testHTML(self):
        """
        Test HTML to Markdown
        """

        # Headings
        self.assertMarkdown("<h1>This is a test</h1>", "# This is a test")
        self.assertMarkdown("<h6>This is a test</h6>", "###### This is a test")

        # Blockquotes
        self.assertMarkdown("<blockquote>This is a test</blockquote>", "> This is a test")

        # Lists
        self.assertMarkdown("<ul><li>Test1</li><li>Test2</li></ul>", "- Test1\n- Test2")
        self.assertMarkdown("<ol><li>Test1</li><li>Test2</li></ol>", "1. Test1\n2. Test2")

        # Code
        self.assertMarkdown("<code>This is a test</code>", "```\nThis is a test\n```")
        self.assertMarkdown("<pre>This is a test</pre>", "```\nThis is a test\n```")

        # Tables
        self.assertMarkdown(
            "<table><tr><th>Header1</th><th>Header2</th></tr><tr><td>Test1</td><td>Test2</td></tr></table>",
            "|Header1|Header2|\n|---|---|\n|Test1|Test2|",
        )

        # Ignore list
        self.assertMarkdown("<aside>This is a test</aside>", "")

        # Text formatting
        self.assertMarkdown("<p>This is a test</p>", "This is a test")
        self.assertMarkdown("<p>This is a <b>test</b</p>", "This is a **test**")
        self.assertMarkdown("<p>This is a <strong>test</strong></p>", "This is a **test**")
        self.assertMarkdown("<p>This is a <i>test</i></p>", "This is a *test*")
        self.assertMarkdown("<p>This is a <em>test</em></p>", "This is a *test*")
        self.assertMarkdown("<p>This is a <a href='link'>test</a>", "This is a [test](link)")

        # Collapse to outer tag
        self.assertMarkdown("<p>This is a <strong><em>test</em></strong></p>", "This is a **test**")
        self.assertMarkdown("<p>This is a <em><strong>test</strong></em></p>", "This is a *test*")

    def testParagraphs(self):
        """
        Test extraction to paragraphs
        """

        textractor = Textractor(paragraphs=True)

        # Extract text as paragraphs
        paragraphs = textractor(Utils.PATH + "/article.pdf")

        # Check number of paragraphs is as expected
        self.assertEqual(len(paragraphs), 11)

    def testSections(self):
        """
        Test extraction to sections
        """

        textractor = Textractor(sections=True)

        # Extract as sections
        sections = textractor(Utils.PATH + "/document.pdf")

        # Check number of sections is as expected
        self.assertEqual(len(sections), 3)

    def testSentences(self):
        """
        Test extraction to sentences
        """

        textractor = Textractor(sentences=True)

        # Extract text as sentences
        sentences = textractor(Utils.PATH + "/article.pdf")

        # Check number of sentences is as expected
        self.assertEqual(len(sentences), 17)

    def testSingle(self):
        """
        Test a single extraction with no tokenization of the results
        """

        textractor = Textractor()

        # Extract text as a single block
        text = textractor(Utils.PATH + "/article.pdf")

        # Check length of text is as expected
        self.assertEqual(len(text), 2471)

    def testTable(self):
        """
        Test table extraction
        """

        textractor = Textractor()

        # Extract text as a single block
        for name in ["document.docx", "spreadsheet.xlsx"]:
            text = textractor(f"{Utils.PATH}/{name}")

            # Check for table header
            self.assertTrue("|---|" in text)

    def testTikaFlag(self):
        """
        Test legacy tika flag
        """

        textractor = Textractor(tika=True)
        self.assertIsNotNone(textractor.html)

        textractor = Textractor(tika=False)
        self.assertIsNone(textractor.html)

    def testURL(self):
        """
        Test parsing a remote URL
        """

        # Test parsing URLs for each backend
        for backend in ["docling", "tika"]:
            textractor = Textractor(backend=backend)
            text = textractor("https://github.com/neuml/txtai")
            self.assertTrue("txtai is an all-in-one AI framework" in text)

    def assertMarkdown(self, html, expected):
        """
        Helper method to assert generated markdown is as expected.

        Args:
            html: input html snippet
            expected: expected markdown text
        """

        textractor = Textractor()
        self.assertEqual(textractor(f"<html><body>{html}</body></html>"), expected)

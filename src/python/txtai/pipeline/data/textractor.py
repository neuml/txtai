"""
Textractor module
"""

import contextlib
import os

from subprocess import Popen
from urllib.request import urlopen

# Conditional import
try:
    from bs4 import BeautifulSoup, NavigableString
    from tika import parser

    TIKA = True
except ImportError:
    TIKA = False

from .segmentation import Segmentation


class Textractor(Segmentation):
    """
    Extracts text from files.
    """

    def __init__(self, sentences=False, lines=False, paragraphs=False, minlength=None, join=False, tika=True, sections=False):
        if not TIKA:
            raise ImportError('Textractor pipeline is not available - install "pipeline" extra to enable')

        super().__init__(sentences, lines, paragraphs, minlength, join, sections)

        # Determine if Tika (default if Java is available) or Beautiful Soup should be used
        # Beautiful Soup only supports HTML, Tika supports a wide variety of file formats, including HTML.
        self.tika = self.checkjava() if tika else False

        # HTML to Text extractor
        self.extract = Extract(self.sections)

    def text(self, text):
        # Use Tika if available
        if self.tika:
            # Format file urls as local file paths
            text = text.replace("file://", "")

            # Parse content to XHTML
            parsed = parser.from_file(text, xmlContent=True)
            text = parsed["content"]
        else:
            # Fallback to XHTML-only support, read data from url/path
            text = f"file://{text}" if os.path.exists(text) else text
            with contextlib.closing(urlopen(text)) as connection:
                text = connection.read()

        # Extract text from HTML
        return self.extract(text)

    def checkjava(self, path=None):
        """
        Checks if a Java executable is available for Tika.

        Args:
            path: path to java executable

        Returns:
            True if Java is available, False otherwise
        """

        # Get path to java executable if path not set
        if not path:
            path = os.environ.get("TIKA_JAVA", "java")

        # pylint: disable=R1732,W0702,W1514
        # Check if java binary is available on path
        try:
            _ = Popen(path, stdout=open(os.devnull, "w"), stderr=open(os.devnull, "w"))
        except:
            return False

        return True


class Extract:
    """
    HTML to Text extractor. A limited set of Markdown is applied for organizing container elements such as tables and lists.
    Visual formatting is not included (bold, italic, styling etc).
    """

    def __init__(self, sections):
        """
        Create a new Extract instance.

        Args:
            sections: True if section parsing enabled, False otherwise
        """

        self.sections = sections

    def __call__(self, html):
        """
        Transforms input HTML into formatted text.

        Args:
            html: input html

        Returns:
            formatted text
        """

        # HTML Parser
        soup = BeautifulSoup(html, features="html.parser")

        # Extract text from each body element
        nodes = []
        for body in soup.find_all("body"):
            nodes.append(self.process(body))

        # Return extracted text, fallback to default text extraction if no nodes found
        return "\n".join(nodes) if nodes else soup.get_text()

    def process(self, node):
        """
        Extracts text from a node. This method applies transforms for containers, tables, lists and text.
        Page breaks are detected and reflected in the output text as a page break character.

        Args:
            node: input node

        Returns:
            node text
        """

        if node.name == "table":
            return self.table(node)
        if node.name in ("ul", "ol"):
            return self.items(node)

        # Get page break symbol, if available
        page = node.name and node.get("class") and "page" in node.get("class")

        # Get node children
        children = self.children(node)

        # Join elements into text
        text = "\n".join(self.process(node) for node in children) if self.iscontainer(node, children) else self.text(node)

        # Add page breaks, if section parsing enabled. Otherwise add node text.
        return f"{text}\f" if page and self.sections else text

    def text(self, node):
        """
        Text handler. This method flattens a node and it's children to text.

        Args:
            node: input node

        Returns:
            node text
        """

        # Get node children if available, otherwise use node as item
        items = self.children(node)
        items = items if items else [node]

        # Join text elements
        text = "".join(x.text for x in items)

        # Return text, strip leading/trailing whitespace if this is a string only node
        return text if node.name else text.strip()

    def table(self, node):
        """
        Table handler. This method transforms a HTML table into a Markdown formatted table.

        Args:
            node: input node

        Returns:
            table as markdown
        """

        elements, header = [], False

        # Process all rows
        rows = node.find_all("tr")
        for row in rows:
            # Get list of columns for row
            columns = row.find_all(lambda tag: tag.name in ("th", "td"))

            # Add columns with separator
            elements.append(f"|{'|'.join(self.process(column) for column in columns)}|")

            # If there are multiple rows, add header format row
            if not header and len(rows) > 1:
                elements.append(f"{'|---' * len(columns)}|")
                header = True

        # Join elements together as string
        return "\n".join(elements)

    def items(self, node):
        """
        List handler. This method transforms a HTML ordered/unordered list into a Markdown formatted list.

        Args:
            node: input node

        Returns:
            list as markdown
        """

        elements = []
        for x, element in enumerate(node.find_all("li")):
            # Unordered lists use dashes. Ordered lists use numbers.
            prefix = "-" if node.name == "ul" else f"{x + 1}."

            # Add list element
            elements.append(f"  {prefix} {self.process(element)}")

        # Join elements together as string
        return "\n".join(elements)

    def iscontainer(self, node, children):
        """
        Analyzes a node and it's children to determine if this is a container element. A container
        element is defined as being a div, body or not having any string elements as children.

        Args:
            node: input node
            nodes: input node's children

        Returns:
            True if this is a container element, False otherwise
        """

        return children and (node.name in ("div", "body") or not any(isinstance(x, NavigableString) for x in children))

    def children(self, node):
        """
        Gets the node children, if available.

        Args:
            node: input node

        Returns:
            node children or None if not available
        """

        if node.name and node.contents:
            # Iterate over children and remove whitespace-only string nodes
            return [node for node in node.contents if node.name or node.text.strip()]

        return None

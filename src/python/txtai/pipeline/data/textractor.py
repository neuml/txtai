"""
Textractor module
"""

import contextlib
import os
import tempfile

from subprocess import Popen
from urllib.parse import urlparse
from urllib.request import urlopen, Request

# Conditional import
try:
    from bs4 import BeautifulSoup, NavigableString
    from tika import detector, parser

    TIKA = True
except ImportError:
    TIKA = False

from .segmentation import Segmentation


class Textractor(Segmentation):
    """
    Extracts text from files.
    """

    def __init__(self, sentences=False, lines=False, paragraphs=False, minlength=None, join=False, tika=True, sections=False, headers=None):
        if not TIKA:
            raise ImportError('Textractor pipeline is not available - install "pipeline" extra to enable')

        super().__init__(sentences, lines, paragraphs, minlength, join, sections)

        # Determine if Apache Tika (default if Java is available) or Beautiful Soup should be used
        # Beautiful Soup only supports HTML, Tika supports a wide variety of file formats.
        self.tika = self.checkjava() if tika else False

        # HTML to Text extractor
        self.extract = Extract(self.paragraphs, self.sections)

        # HTTP headers
        self.headers = headers if headers else {}

    def text(self, text):
        # Check if text is a valid file path or url
        path, exists = self.valid(text)

        if not path:
            # Not a valid file path, treat input as data
            html = text

        elif self.tika:
            # Use Tika if available
            # Retrieve remote file, if necessary
            path = path if exists else self.download(path)

            # Parse content to XHTML
            html = self.html(path)

            # Delete temporary file
            if not exists:
                os.remove(path)

        else:
            # Read data from url/path
            html = self.retrieve(path)

        # Extract text from HTML
        return self.extract(html)

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

    def valid(self, path):
        """
        Checks if path is a valid local file or web url. Returns path if valid along with a flag
        denoting if the path exists locally.

        Args:
            path: path to check

        Returns:
            (path, exists)
        """

        # Convert file urls to local paths
        path = path.replace("file://", "")

        # Check if this is a local file path or local file url
        exists = os.path.exists(path)

        # Consider local files and HTTP urls valid
        return (path if exists or urlparse(path).scheme in ("http", "https") else None, exists)

    def download(self, url):
        """
        Downloads content of url to a temporary file.

        Args:
            url: input url

        Returns:
            temporary file path
        """

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as output:
            path = output.name

            # Retrieve and write data to temporary file
            output.write(self.retrieve(url))

        return path

    def retrieve(self, url):
        """
        Retrieves content from url.

        Args:
            url: input url

        Returns:
            data
        """

        # Local file
        if os.path.exists(url):
            with open(url, "rb") as f:
                return f.read()

        # Remote file
        with contextlib.closing(urlopen(Request(url, headers=self.headers))) as connection:
            return connection.read()

    def html(self, path):
        """
        Parses content to HTML using Apache Tika.

        Args:
            path: file path

        Returns:
            html
        """

        # Skip if file is already HTML
        mimetype = detector.from_file(path)
        if mimetype in ("text/html", "text/xhtml"):
            return self.retrieve(path)

        # Parse content to XHTML
        parsed = parser.from_file(path, xmlContent=True)
        return parsed["content"]


class Extract:
    """
    HTML to Text extractor. A limited set of Markdown is applied for organizing container elements such as tables and lists.
    Visual formatting is not included (bold, italic, styling etc).
    """

    def __init__(self, paragraphs, sections):
        """
        Create a new Extract instance.

        Args:
            paragraphs: True if paragraph parsing enabled, False otherwise
            sections: True if section parsing enabled, False otherwise
        """

        self.paragraphs = paragraphs
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

        # Ignore script and style tags
        for script in soup.find_all(["script", "style"]):
            script.decompose()

        # Check if articles are embedded in this html
        article = soup.find("article") is not None

        # Extract text from each section element
        nodes = []
        for node in soup.find_all("article" if article else "body"):
            nodes.append(self.process(node, article))

        # Return extracted text, fallback to default text extraction if no nodes found
        return "\n".join(self.metadata(soup) + nodes) if nodes else soup.get_text()

    def process(self, node, article):
        """
        Extracts text from a node. This method applies transforms for containers, tables, lists and text.
        Page breaks are detected and reflected in the output text as a page break character.

        Args:
            node: input node
            article: True if the main section node is an article

        Returns:
            node text
        """

        if node.name == "table":
            return self.table(node, article)
        if node.name in ("ul", "ol"):
            return self.items(node, article)

        # Get page break symbol, if available
        page = node.name and node.get("class") and "page" in node.get("class")

        # Get node children
        children = self.children(node)

        # Join elements into text
        if self.iscontainer(node, children):
            texts = [self.process(node, article) for node in children]
            text = "\n".join(text for text in texts if text or not article)
        else:
            text = self.text(node, article)

        # Add page breaks, if section parsing enabled. Otherwise add node text.
        return f"{text}\f" if page and self.sections else text

    def metadata(self, node):
        """
        Builds a metadata section. The metadata section consists of the title and
        description fields.

        Args:
            node: input document node

        Returns:
            metadata as a list
        """

        title = node.find("title")
        metadata = [title.text] if title else []

        description = node.find("meta", attrs={"name": "description"})
        if description and description["content"]:
            metadata.append(f"\n{description['content']}")

        # Add separator
        if metadata:
            metadata.append("\f" if self.sections else "\n\n")

        return metadata

    def text(self, node, article):
        """
        Text handler. This method flattens a node and it's children to text.

        Args:
            node: input node
            article: True if the main section node is an article

        Returns:
            node text
        """

        # Get node children if available, otherwise use node as item
        items = self.children(node)
        items = items if items else [node]

        # Join text elements
        text = "".join(x.text for x in items)

        # Article text processing
        text = self.articletext(node, text) if article else text

        # Return text, strip leading/trailing whitespace if this is a string only node
        text = text if node.name and text else text.strip()

        return text

    def table(self, node, article):
        """
        Table handler. This method transforms a HTML table into a Markdown formatted table.

        Args:
            node: input node
            article: True if the main section node is an article

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
            elements.append(f"|{'|'.join(self.process(column, article) for column in columns)}|")

            # If there are multiple rows, add header format row
            if not header and len(rows) > 1:
                elements.append(f"{'|---' * len(columns)}|")
                header = True

        # Join elements together as string
        return "\n".join(elements)

    def items(self, node, article):
        """
        List handler. This method transforms a HTML ordered/unordered list into a Markdown formatted list.

        Args:
            node: input node
            article: True if the main section node is an article

        Returns:
            list as markdown
        """

        elements = []
        for x, element in enumerate(node.find_all("li")):
            # Unordered lists use dashes. Ordered lists use numbers.
            prefix = "-" if node.name == "ul" else f"{x + 1}."

            # List item text
            text = self.process(element, article)

            # Add list element
            if text:
                elements.append(f"  {prefix} {text}")

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

        return children and (node.name in ("div", "body", "article") or not any(isinstance(x, NavigableString) for x in children))

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

    def articletext(self, node, text):
        """
        Transforms node text using article parsing rules. Article parsing is designed to extract text content from web articles.
        It ignores navigation headers and other superfluous elements.

        Args:
            node: input node
            text: current text

        Returns:
            article text
        """

        # Check if text is valid article text
        text = text if (node.name in ["p", "th", "td", "li", "a"] or self.isheader(node)) and not self.islink(node) else ""
        if text:
            # Replace non-breaking space with newline
            text = text.replace("\xa0", "\n")

            # Format paragraph whitespace
            if node.name == "p":
                text = f"{text.strip()}\n\n" if self.paragraphs else f"{text.strip()}\n"

            # Format header whitespace
            if self.sections and self.isheader(node):
                text = f"\f{text.strip()}\n"

        return text

    def isheader(self, node):
        """
        Checks if node is a header node.

        Args:
            node: input node

        Returns:
            True if node is a header node, False otherwise
        """

        return node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]

    def islink(self, node):
        """
        Checks if node is a link node. This method does not consider links without tables as link nodes.

        Args:
            node: input node

        Returns:
            True if node is a link node, False otherwise
        """

        # Check if this is a link node or link container
        link, parent = False, node
        while parent:
            if parent.name == "a":
                link = True
                break

            parent = parent.parent

        # Return if this node or any parents are a link. Ignore links in table cells.
        return link and node.parent.name not in ("th", "td")

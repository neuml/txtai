"""
Console module tests
"""

import contextlib
import io
import os
import tempfile
import unittest

from txtai.console import Console
from txtai.embeddings import Embeddings

APPLICATION = """
path: %s

workflow:
  test:
     tasks:
       - task: console
"""


class TestConsole(unittest.TestCase):
    """
    Console tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        # Create embeddings model, backed by sentence-transformers & transformers
        cls.embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})

        # Create an index for the list of text
        cls.embeddings.index([(uid, text, None) for uid, text in enumerate(cls.data)])

        # Create app paths
        cls.apppath = os.path.join(tempfile.gettempdir(), "console.yml")
        cls.embedpath = os.path.join(tempfile.gettempdir(), "embeddings.console")

        # Create app.yml
        with open(cls.apppath, "w", encoding="utf-8") as out:
            out.write(APPLICATION % cls.embedpath)

        # Save index as uncompressed and compressed
        cls.embeddings.save(cls.embedpath)
        cls.embeddings.save(f"{cls.embedpath}.tar.gz")

        # Create console
        cls.console = Console(cls.embedpath)

    def testApplication(self):
        """
        Test application
        """

        self.assertNotIn("Traceback", self.command(f".load {self.apppath}"))
        self.assertIn("1", self.command(".limit 1"))
        self.assertIn("Maine man wins", self.command("feel good story"))

    def testConfig(self):
        """
        Test .config command
        """

        self.assertIn("tasks", self.command(".config"))

    def testEmbeddings(self):
        """
        Test embeddings index
        """

        self.assertNotIn("Traceback", self.command(f".load {self.embedpath}.tar.gz"))
        self.assertNotIn("Traceback", self.command(f".load {self.embedpath}"))
        self.assertIn("1", self.command(".limit 1"))
        self.assertIn("Maine man wins", self.command("feel good story"))

    def testEmbeddingsNoDatabase(self):
        """
        Test embeddings with no database/content
        """

        console = Console()

        # Create embeddings model, backed by sentence-transformers & transformers
        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})

        # Create an index for the list of text
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Set embeddings on console
        console.app = embeddings
        self.assertIn("4", self.command("feel good story", console))

    def testEmpty(self):
        """
        Test empty console instance
        """

        console = Console()
        self.assertIn("AttributeError", self.command("search", console))

    def testHighlight(self):
        """
        Test .highlight command
        """

        self.assertIn("highlight", self.command(".highlight"))
        self.assertIn("wins", self.command("feel good story"))
        self.assertIn("Taiwan", self.command("asia"))

    def testPreloop(self):
        """
        Test preloop
        """

        self.assertIn("txtai console", self.preloop())

    def testWorkflow(self):
        """
        Test .workflow command
        """

        self.command(f".load {self.apppath}")
        self.assertIn("echo", self.command(".workflow test echo"))

    def command(self, command, console=None):
        """
        Runs a console command.

        Args:
            command: command to run
            console: console instance, defaults to self.console

        Returns:
            command output
        """

        # Run info
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            if not console:
                console = self.console

            console.onecmd(command)

        return output.getvalue()

    def preloop(self):
        """
        Runs console.preloop and redirects stdout.

        Returns:
            preloop output
        """

        # Run info
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.console.preloop()

        return output.getvalue()

"""
Application module tests
"""

import os
import unittest
import types

from unittest.mock import patch

from txtai.app import Application
from txtai.embeddings import Documents, Embeddings
from txtai.pipeline import Pipeline


class TestApp(unittest.TestCase):
    """
    Application tests.
    """

    @patch("txtai.app.base.Documents", wraps=Documents)
    def testAddEmpty(self, documents):
        """
        Test adding an empty batch before a batch with data
        """

        app = Application(
            """
            writable: true
            embeddings:
                keyword: true
                content: true
        """
        )

        app.add([])
        app.add([{"id": 0, "text": "hello world"}])

        # Both batches share one documents queue, no temporary file is left behind
        self.assertEqual(documents.call_count, 1)

        app.index()
        self.assertEqual(app.search("hello")[0]["text"], "hello world")

    def testConfig(self):
        """
        Test a file not found config exception
        """

        with self.assertRaises(FileNotFoundError):
            Application.read("No file here")

    def testBuiltinFunction(self):
        """
        Test zero-argument builtin SQL functions with direct and application configuration
        """

        for application in (False, True):
            with self.subTest(application=application):
                if application:
                    embeddings = Application(
                        """
                        embeddings:
                            defaults: false
                            content: true
                            functions:
                                - {name: processid, function: os.getpid, argcount: 0}
                        """
                    ).embeddings
                else:
                    embeddings = Embeddings(defaults=False, content=True, functions=[{"name": "processid", "function": "os.getpid", "argcount": 0}])

                try:
                    embeddings.index(["note"])
                    self.assertEqual(embeddings.search("select processid() pid from txtai")[0]["pid"], os.getpid())
                finally:
                    embeddings.close()

    def testParameter(self):
        """
        Test resolving application parameter
        """

        app = Application(
            """
            testapp.TestPipeline:
                application:
        """
        )

        # Check that application instance is not None
        self.assertIsNotNone(app.pipelines["testapp.TestPipeline"].application)

    def testStream(self):
        """
        Test workflow streams
        """

        app = Application(
            """
            workflow:
                stream:
                    stream:
                        action: testapp.TestStream
                    tasks:
                        - nop
                batchstream:
                    stream:
                        action: testapp.TestStream
                        batch: True
                    tasks:
                        - nop
        """
        )

        def generator():
            yield 10

        # Test single stream
        self.assertEqual(list(app.workflow("stream", [10])), list(range(10)))

        # Test batch stream
        self.assertEqual(list(app.workflow("batchstream", generator())), list(range(10)))


class TestPipeline(Pipeline):
    """
    Test pipeline with an application parameter.
    """

    def __init__(self, application):
        self.application = application


class TestStream:
    """
    Test workflow stream
    """

    def __call__(self, arg):
        if isinstance(arg, types.GeneratorType):
            for x in arg:
                yield from range(int(x))
        else:
            yield from range(int(arg))

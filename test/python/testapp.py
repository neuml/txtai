"""
Application module tests
"""

import tempfile
import unittest
import types

from unittest.mock import patch

from txtai.app import Application
from txtai.pipeline import Pipeline


class TestApp(unittest.TestCase):
    """
    Application tests.
    """

    def testConfig(self):
        """
        Test a file not found config exception
        """

        with self.assertRaises(FileNotFoundError):
            Application.read("No file here")

    def testEmptyAdd(self):
        """
        Test an empty first add keeps a single documents queue
        """

        app = Application("writable: true\nembeddings:\n  keyword: true\n  content: true\n")

        with patch("tempfile.NamedTemporaryFile", wraps=tempfile.NamedTemporaryFile) as queue:
            app.add([])
            app.add([{"id": 0, "text": "apple"}])
            app.index()

        self.assertEqual(len([x for x in queue.call_args_list if x.kwargs.get("suffix") == ".docs"]), 1)
        self.assertEqual(app.search("apple")[0]["id"], "0")

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

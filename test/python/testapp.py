"""
Application module tests
"""

import os
import unittest
import types

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock, patch

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

    def testConcurrentFlush(self):
        """
        A caller waiting for the lock can find its document queue already flushed.
        """

        for operation in ("index", "upsert"):
            with self.subTest(operation=operation):
                self.checkConcurrentFlush(operation)

    def checkConcurrentFlush(self, operation):
        """Check two callers flushing the same queue."""

        app = Application({"writable": True, "embeddings": {"keyword": True, "content": True}})
        app.add([{"id": "doc", "text": "apple"}])
        waiting = Event()
        lock = app.lock

        def acquire():
            waiting.set()
            return lock.acquire()

        # Hold the real lock until the second caller is ready to flush.
        with ThreadPoolExecutor(max_workers=1) as pool, patch.object(app, "lock", MagicMock()) as gate:
            gate.__enter__.side_effect = acquire
            gate.__exit__.side_effect = lambda *_: lock.release()
            with lock:
                flush = getattr(app, operation)
                future = pool.submit(flush)
                self.assertTrue(waiting.wait(5))
                flush()
            future.result(timeout=5)

        self.assertEqual(app.search("apple")[0]["id"], "doc")
        app.embeddings.close()

    def testConfig(self):
        """
        Test a file not found config exception
        """

        with self.assertRaises(FileNotFoundError):
            Application.read("No file here")

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

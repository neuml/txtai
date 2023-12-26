"""
Extension module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from fastapi import APIRouter
from fastapi.testclient import TestClient

from txtai.api import application, Extension
from txtai.pipeline import Pipeline

# Example pipeline extension
PIPELINES = """
testapi.testextension.SamplePipeline:
"""


class SampleRouter:
    """
    Sample API router.
    """

    router = APIRouter()

    @staticmethod
    @router.get("/sample")
    def sample(text: str):
        """
        Calls sample pipeline.

        Args:
            text: input text

        Returns:
            formatted text
        """

        return application.get().pipeline("testapi.testextension.SamplePipeline", (text,))


class SampleExtension(Extension):
    """
    Sample API extension.
    """

    def __call__(self, app):
        app.include_router(SampleRouter().router)


class SamplePipeline(Pipeline):
    """
    Sample pipeline.
    """

    def __call__(self, text):
        return text.lower()


class TestExtension(unittest.TestCase):
    """
    API tests for extensions.
    """

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"),
            "API_CLASS": "txtai.api.API",
            "EXTENSIONS": "testapi.testextension.SampleExtension",
        },
    )
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w", encoding="utf-8") as output:
            output.write(PIPELINES)

        # Create new application and set on client
        application.app = application.create()
        client = TestClient(application.app)
        application.start()

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class.
        """

        cls.client = TestExtension.start()

    def testEmpty(self):
        """
        Test an empty extension
        """

        extension = Extension()
        self.assertIsNone(extension(None))

    def testExtension(self):
        """
        Test a pipeline extension
        """

        text = self.client.get("sample?text=Test%20String").json()
        self.assertEqual(text, "test string")

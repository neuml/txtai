"""
Application module tests
"""

import unittest

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


class TestPipeline(Pipeline):
    """
    Test pipeline with an application parameter.
    """

    def __init__(self, application):
        self.application = application

"""
Application module tests
"""

import unittest

from txtai.app import Application


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

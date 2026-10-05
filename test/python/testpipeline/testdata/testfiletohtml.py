"""
FileToHTML module tests
"""

import os
import unittest

from unittest.mock import patch

from txtai.pipeline.data.filetohtml import Tika


class TestFileToHTML(unittest.TestCase):
    """
    FileToHTML tests.
    """

    @patch.dict(os.environ, {"TIKA_JAVA": "1112444abc"})
    def testTika(self):
        """
        Test the Tika.available returns False when Java is not available
        """

        self.assertFalse(Tika.available())

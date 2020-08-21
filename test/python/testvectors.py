"""
Vectors module tests
"""

import os
import tempfile
import unittest

from txtai.vectors import WordVectors

class TestVectors(unittest.TestCase):
    """
    Vectors tests
    """

    def testBuild(self):
        """
        Test a WordVectors build.
        """

        # Build word vectors on README file
        WordVectors.build("README.md", 300, 3, os.path.join(tempfile.gettempdir(), "vectors"))

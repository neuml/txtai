"""
Zvec ANN module tests
"""

from txtai.ann import ANNFactory

from .base import DenseTest


class TestZvec(DenseTest):
    """
    Zvec ANN tests.
    """

    def testZvec(self):
        """
        Test zvec backend
        """

        self.runTests("zvec")
        self.deletenumpy("zvec")

    def testZvecCustom(self):
        """
        Test zvec backend with custom settings
        """

        self.runTests("zvec", {"zvec": {"m": 16}})

        # Test invalid file path handled
        with self.assertRaises(FileNotFoundError):
            ann = ANNFactory.create({"backend": "zvec"})
            ann.load("non-exist-path")

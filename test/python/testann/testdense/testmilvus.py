"""
Milvus ANN module tests
"""

from txtai.ann import ANNFactory

from .base import DenseTest


class TestMilvus(DenseTest):
    """
    Milvus ANN tests.
    """

    def testMilvus(self):
        """
        Test milvus-lite backend
        """

        self.runTests("milvus")

    def testMilvusCustom(self):
        """
        Test milvus-lite backend with custom settings
        """

        self.runTests("milvus", {"milvus": {"m": 16}})

        # Test invalid file path handled
        with self.assertRaises(FileNotFoundError):
            ann = ANNFactory.create({"backend": "milvus"})
            ann.load("non-exist-path")

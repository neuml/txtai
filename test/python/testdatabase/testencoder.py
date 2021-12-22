"""
Tests encoding/decoding database objects
"""

import glob
import unittest

from PIL import Image

from txtai.embeddings import Embeddings

# pylint: disable = C0411
from utils import Utils


class TestEncoder(unittest.TestCase):
    """
    Encoder tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = []
        for path in glob.glob(Utils.PATH + "/*jpg"):
            cls.data.append((path, {"object": Image.open(path)}, None))

        # Create embeddings model, backed by sentence-transformers & transformers
        cls.embeddings = Embeddings(
            {"method": "sentence-transformers", "path": "sentence-transformers/clip-ViT-B-32", "content": True, "objects": "image"}
        )

    def testDefault(self):
        """
        Tests an index with default encoder
        """

        try:
            # Change to use default encoder
            self.embeddings.config["objects"] = True
            data = [(0, {"object": bytearray([1, 2, 3]), "text": "default test"}, None)]

            # Create an index for the list of images
            self.embeddings.index(data)

            result = self.embeddings.search("select object from txtai limit 1")[0]

            self.assertEqual(result["object"].getvalue(), bytearray([1, 2, 3]))
        finally:
            self.embeddings.config["objects"] = "image"

    def testImages(self):
        """
        Tests an index with image encoder
        """

        # Create an index for the list of images
        self.embeddings.index(self.data)

        result = self.embeddings.search("select id, object from txtai where similar('universe') limit 1")[0]

        self.assertTrue(result["id"].endswith("stars.jpg"))
        self.assertTrue(isinstance(result["object"], Image.Image))

    def testPickle(self):
        """
        Tests an index with pickle encoder
        """

        try:
            # Change to use default encoder
            self.embeddings.config["objects"] = "pickle"
            data = [(0, {"object": [1, 2, 3, 4, 5], "text": "default test"}, None)]

            # Create an index for the list of images
            self.embeddings.index(data)

            result = self.embeddings.search("select object from txtai limit 1")[0]

            self.assertEqual(result["object"], [1, 2, 3, 4, 5])
        finally:
            self.embeddings.config["objects"] = "image"

    def testReindex(self):
        """
        Tests reindex with objects
        """

        # Create an index for the list of images
        self.embeddings.index(self.data)

        # Reindex images
        self.embeddings.reindex({"method": "sentence-transformers", "path": "sentence-transformers/clip-ViT-B-32"})

        result = self.embeddings.search("select id, object from txtai where similar('universe') limit 1")[0]

        self.assertTrue(result["id"].endswith("stars.jpg"))
        self.assertTrue(isinstance(result["object"], Image.Image))

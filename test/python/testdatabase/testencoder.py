"""
Test encoding/decoding database objects
"""

import glob
import os
import unittest
import tempfile

from io import BytesIO

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

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup data.
        """

        if cls.embeddings:
            cls.embeddings.close()

    def testDefault(self):
        """
        Test an index with default encoder
        """

        try:
            # Set default encoder
            self.embeddings.config["objects"] = True

            # Test all database providers
            for content in ["duckdb", "sqlite"]:
                self.embeddings.config["content"] = content

                data = [(0, {"object": bytearray([1, 2, 3]), "text": "default test"}, None)]

                # Create an index
                self.embeddings.index(data)

                result = self.embeddings.search("select object from txtai limit 1")[0]

                self.assertEqual(result["object"].getvalue(), bytearray([1, 2, 3]))
        finally:
            self.embeddings.config["objects"] = "image"
            self.embeddings.config["content"] = True

    def testImages(self):
        """
        Test an index with image encoder
        """

        # Create an index for the list of images
        self.embeddings.index(self.data)

        result = self.embeddings.search("select id, object from txtai where similar('universe') limit 1")[0]

        self.assertTrue(result["id"].endswith("stars.jpg"))
        self.assertTrue(isinstance(result["object"], Image.Image))

    def testPickle(self):
        """
        Test an index with pickle encoder
        """

        try:
            # Set pickle encoder
            self.embeddings.config["objects"] = "pickle"
            data = [(0, {"object": [1, 2, 3, 4, 5], "text": "default test"}, None)]

            # Create an index
            self.embeddings.index(data)

            result = self.embeddings.search("select object from txtai limit 1")[0]

            self.assertEqual(result["object"], [1, 2, 3, 4, 5])
        finally:
            self.embeddings.config["objects"] = "image"

    def testReindex(self):
        """
        Test reindex with objects
        """

        # Create an index for the list of images
        self.embeddings.index(self.data)

        # Reindex images
        self.embeddings.reindex({"method": "sentence-transformers", "path": "sentence-transformers/clip-ViT-B-32"})

        result = self.embeddings.search("select id, object from txtai where similar('universe') limit 1")[0]

        self.assertTrue(result["id"].endswith("stars.jpg"))
        self.assertTrue(isinstance(result["object"], Image.Image))

    def testReindexFunction(self):
        """
        Test reindex with objects and a function
        """

        try:
            # Streaming function that loads images on the fly
            def prepare(documents):
                for uid, data, tags in documents:
                    yield (uid, Image.open(data), tags)

            # Create an index for the list of images
            self.embeddings.index(self.data)

            # Set default encoder and use function to load images
            self.embeddings.config["objects"] = True

            # Save and load index to force default encoder
            index = os.path.join(tempfile.gettempdir(), "objects")
            self.embeddings.save(index)
            self.embeddings.load(index)

            # Reindex images
            self.embeddings.reindex({"method": "sentence-transformers", "path": "sentence-transformers/clip-ViT-B-32"}, function=prepare)

            result = self.embeddings.search("select id, object from txtai where similar('universe') limit 1")[0]

            self.assertTrue(result["id"].endswith("stars.jpg"))
            self.assertTrue(isinstance(result["object"], BytesIO))
        finally:
            self.embeddings.config["objects"] = "image"

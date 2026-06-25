"""
TwelveLabs module tests
"""

import os
import unittest

from unittest.mock import MagicMock, patch

import numpy as np

from txtai.vectors import VectorsFactory


class TestTwelveLabs(unittest.TestCase):
    """
    TwelveLabs Marengo vectors tests
    """

    def mock(self, dimensions=512):
        """
        Builds a mock TwelveLabs client returning a fixed-size text embedding.

        Args:
            dimensions: embedding dimensions

        Returns:
            mock client
        """

        client = MagicMock()
        segment = MagicMock()
        segment.float_ = [0.1] * dimensions
        client.embed.create.return_value.text_embedding.segments = [segment]
        return client

    @patch("txtai.vectors.dense.twelvelabs.TwelveLabsAPI")
    def testIndex(self, api):
        """
        Test indexing with TwelveLabs Marengo vectors
        """

        # Mock client
        api.return_value = self.mock()

        # TwelveLabs vectors instance - method inferred from marengo path prefix
        model = VectorsFactory.create({"path": "marengo3.0"}, None)

        ids, dimension, batches, stream = model.index([(0, "a red car driving fast", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 512)
        self.assertEqual(batches, 1)
        self.assertTrue(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 512))

    @patch("txtai.vectors.dense.twelvelabs.TwelveLabsAPI")
    def testImageModality(self, api):
        """
        Test that an image_url input routes to the image embedding
        """

        client = MagicMock()
        segment = MagicMock()
        segment.float_ = [0.2] * 512
        client.embed.create.return_value.image_embedding.segments = [segment]
        api.return_value = client

        model = VectorsFactory.create({"path": "marengo3.0"}, None)
        embedding = model.transform((0, {"image_url": "https://example.com/cat.jpg"}, None))

        # Verify the image embedding was used and create() received the image_url
        self.assertEqual(embedding.shape, (512,))
        self.assertEqual(client.embed.create.call_args.kwargs["image_url"], "https://example.com/cat.jpg")

    @unittest.skipIf(not os.environ.get("TWELVELABS_API_KEY"), "TWELVELABS_API_KEY not set")
    def testLive(self):
        """
        Live test - requires a TwelveLabs API key
        """

        model = VectorsFactory.create({"path": "marengo3.0"}, None)
        embedding = model.transform((0, "a red car driving fast", None))

        self.assertEqual(embedding.shape, (512,))

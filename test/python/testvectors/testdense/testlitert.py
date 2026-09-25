"""
LiteRT module tests
"""

import os
import unittest

from unittest.mock import patch

import numpy as np

from huggingface_hub import hf_hub_download
from txtai.vectors import VectorsFactory


class TestLiteRT(unittest.TestCase):
    """
    LiteRT vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create LiteRT instance.
        """

        cls.model = VectorsFactory.create(
            {"path": "neuml/bert-hash-nano-embeddings-litert/bert-hash-nano-embeddings-int4.tflite", "gpu": False}, None
        )

    def testIndex(self):
        """
        Test indexing with LiteRT vectors
        """

        ids, dimension, batches, stream = self.model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 128)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 128))

    @patch("huggingface_hub.hf_hub_download")
    def testTokenizerRepo(self, download):
        """
        Test that a tokenizer at the root of a HF Hub repo is found for a model stored in a subdirectory
        """

        def filedownload(**kwargs):
            # Serve the model from a subdirectory that doesn't have a tokenizer
            if kwargs["filename"] == "litert/tokenizer.json":
                raise FileNotFoundError

            return hf_hub_download(repo_id=kwargs["repo_id"], filename=kwargs["filename"].replace("litert/", ""))

        download.side_effect = filedownload

        model = VectorsFactory.create(
            {"path": "neuml/bert-hash-nano-embeddings-litert/litert/bert-hash-nano-embeddings-int4.tflite", "gpu": False}, None
        )
        self.assertEqual(model.encode(["test"]).shape, (1, 128))

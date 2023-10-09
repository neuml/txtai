"""
Models module tests
"""

import unittest

from unittest.mock import patch

import torch

from txtai.models import Models, ClsPooling, MeanPooling, PoolingFactory


class TestModels(unittest.TestCase):
    """
    Models tests.
    """

    @patch("torch.cuda.is_available")
    def testDeviceid(self, cuda):
        """
        Test the deviceid method
        """

        cuda.return_value = True
        self.assertEqual(Models.deviceid(True), 0)
        self.assertEqual(Models.deviceid(False), -1)
        self.assertEqual(Models.deviceid(0), 0)
        self.assertEqual(Models.deviceid(1), 1)

        # Test direct torch device
        # pylint: disable=E1101
        self.assertEqual(Models.deviceid(torch.device("cpu")), torch.device("cpu"))

        cuda.return_value = False
        self.assertEqual(Models.deviceid(True), -1)
        self.assertEqual(Models.deviceid(False), -1)
        self.assertEqual(Models.deviceid(0), -1)
        self.assertEqual(Models.deviceid(1), -1)

    def testDevice(self):
        """
        Tests the device method
        """

        # pylint: disable=E1101
        self.assertEqual(Models.device("cpu"), torch.device("cpu"))
        self.assertEqual(Models.device(torch.device("cpu")), torch.device("cpu"))

    def testPooling(self):
        """
        Tests pooling methods
        """

        # Device id
        device = Models.deviceid(True)

        # Test mean pooling
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": device})
        self.assertEqual(type(pooling), MeanPooling)

        pooling = PoolingFactory.create({"method": "meanpooling", "path": "flax-sentence-embeddings/multi-qa_v1-MiniLM-L6-cls_dot", "device": device})
        self.assertEqual(type(pooling), MeanPooling)

        # Test CLS pooling
        pooling = PoolingFactory.create({"path": "flax-sentence-embeddings/multi-qa_v1-MiniLM-L6-cls_dot", "device": device})
        self.assertEqual(type(pooling), ClsPooling)

        pooling = PoolingFactory.create({"method": "clspooling", "path": "sentence-transformers/nli-mpnet-base-v2", "device": device})
        self.assertEqual(type(pooling), ClsPooling)

        # Test CLS pooling encoding
        self.assertEqual(pooling.encode(["test"])[0].shape, (768,))

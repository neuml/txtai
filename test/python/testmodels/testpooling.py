"""
Pooling module tests
"""

import unittest

from txtai.models import Models, ClsPooling, MeanPooling, PoolingFactory


class TestPooling(unittest.TestCase):
    """
    Pooling tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize device
        """

        # Device id
        cls.device = Models.deviceid(True)

    def testCLS(self):
        """
        Test CLS pooling
        """

        # Test CLS pooling
        pooling = PoolingFactory.create({"path": "flax-sentence-embeddings/multi-qa_v1-MiniLM-L6-cls_dot", "device": self.device})
        self.assertEqual(type(pooling), ClsPooling)

        pooling = PoolingFactory.create({"method": "clspooling", "path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device})
        self.assertEqual(type(pooling), ClsPooling)

        # Test CLS pooling encoding
        self.assertEqual(pooling.encode(["test"])[0].shape, (768,))

    def testLength(self):
        """
        Test pooling with max_seq_length
        """

        # Test reading max_seq_length parmaeter
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device, "maxlength": True})
        self.assertEqual(pooling.maxlength, 75)

        # Test specified maxlength
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device, "maxlength": 256})
        self.assertEqual(pooling.maxlength, 256)

        # Test max_seq_length is ignored when parameter is omitted
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device})
        self.assertEqual(pooling.maxlength, 512)

        # Test maxlength when max_seq_length not present
        pooling = PoolingFactory.create({"path": "hf-internal-testing/tiny-random-gpt2", "device": self.device, "maxlength": True})
        self.assertEqual(pooling.maxlength, 1024)

    def testMean(self):
        """
        Test mean pooling
        """

        # Test mean pooling
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device})
        self.assertEqual(type(pooling), MeanPooling)

        pooling = PoolingFactory.create(
            {"method": "meanpooling", "path": "flax-sentence-embeddings/multi-qa_v1-MiniLM-L6-cls_dot", "device": self.device}
        )
        self.assertEqual(type(pooling), MeanPooling)

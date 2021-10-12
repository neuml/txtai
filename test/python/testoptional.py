"""
Optional module tests
"""

import unittest

import txtai.ann.factory

from txtai.ann import ANNFactory
from txtai.models import OnnxModel
from txtai.pipeline import HFOnnx, MLOnnx, Segmentation, Textractor, Transcription, Translation
from txtai.vectors import VectorsFactory
from txtai.workflow.task.image import ImageTask
from txtai.workflow.task.storage import StorageTask


class TestOptional(unittest.TestCase):
    """
    Optional tests
    """

    @staticmethod
    def toggle():
        """
        Toggles parameters used to determine the presence of optional libraries
        """

        txtai.ann.factory.ANNOY = not txtai.ann.factory.ANNOY
        txtai.ann.factory.HNSWLIB = not txtai.ann.factory.HNSWLIB

        txtai.models.onnx.ONNX_RUNTIME = not txtai.models.onnx.ONNX_RUNTIME

        txtai.pipeline.hfonnx.ONNX_RUNTIME = not txtai.pipeline.hfonnx.ONNX_RUNTIME
        txtai.pipeline.mlonnx.ONNX_MLTOOLS = not txtai.pipeline.mlonnx.ONNX_MLTOOLS
        txtai.pipeline.segmentation.NLTK = not txtai.pipeline.segmentation.NLTK
        txtai.pipeline.textractor.TIKA = not txtai.pipeline.textractor.TIKA
        txtai.pipeline.transcription.SOUNDFILE = not txtai.pipeline.transcription.SOUNDFILE
        txtai.pipeline.translation.FASTTEXT = not txtai.pipeline.translation.FASTTEXT

        txtai.vectors.factory.WORDS = not txtai.vectors.factory.WORDS
        txtai.vectors.transformers.SENTENCE_TRANSFORMERS = not txtai.vectors.transformers.SENTENCE_TRANSFORMERS

        txtai.workflow.task.image.PIL = not txtai.workflow.task.image.PIL
        txtai.workflow.task.storage.LIBCLOUD = not txtai.workflow.task.storage.LIBCLOUD

    @classmethod
    def setUpClass(cls):
        """
        Simulate optional packages not being installed
        """

        # Toggle parameters
        TestOptional.toggle()

    @classmethod
    def tearDownClass(cls):
        """
        Reset global parameters
        """

        # Toggle parameters back
        TestOptional.toggle()

    def testAnn(self):
        """
        Test missing ann dependencies
        """

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "annoy"})

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "hnsw"})

    def testModel(self):
        """
        Test missing model dependencies
        """

        with self.assertRaises(ImportError):
            OnnxModel(None)

    def testPipeline(self):
        """
        Test missing pipeline dependencies
        """

        with self.assertRaises(ImportError):
            HFOnnx()("google/bert_uncased_L-2_H-128_A-2", quantize=True)

        with self.assertRaises(ImportError):
            MLOnnx()

        with self.assertRaises(ImportError):
            Segmentation()

        with self.assertRaises(ImportError):
            Textractor()

        with self.assertRaises(ImportError):
            Transcription()

        with self.assertRaises(ImportError):
            Translation().detect(["test"])

    def testVectors(self):
        """
        Test missing vectors dependencies
        """

        with self.assertRaises(ImportError):
            VectorsFactory.create({"method": "words"}, None)

        with self.assertRaises(ImportError):
            VectorsFactory.create({"method": "sentence-transformers", "path": ""}, None)

    def testWorkflow(self):
        """
        Test missing workflow dependencies
        """

        with self.assertRaises(ImportError):
            ImageTask()

        with self.assertRaises(ImportError):
            StorageTask()

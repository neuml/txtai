"""
TextToSpeech module tests
"""

import os
import unittest

from unittest.mock import patch

from txtai.pipeline import TextToSpeech


class TestTextToSpeech(unittest.TestCase):
    """
    TextToSpeech tests.
    """

    @unittest.skipIf(os.name == "nt", "testTextToSpeech skipped on Windows")
    def testTextToSpeech(self):
        """
        Test generating speech for text
        """

        tts = TextToSpeech()

        # Check that data is generated
        self.assertGreater(len(tts("This is a test")), 0)

    @unittest.skipIf(os.name == "nt", "testProviders skipped on Windows")
    @patch("onnxruntime.get_available_providers")
    @patch("torch.cuda.is_available")
    def testProviders(self, cuda, providers):
        """
        Test that GPU provider is detected
        """

        # Test CUDA and onnxruntime-gpu installed
        cuda.return_value = True
        providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        tts = TextToSpeech()
        self.assertEqual(tts.providers()[0][0], "CUDAExecutionProvider")

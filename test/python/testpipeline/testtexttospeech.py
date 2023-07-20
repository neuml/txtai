"""
TextToSpeech module tests
"""

import unittest

from unittest.mock import patch

from txtai.pipeline import TextToSpeech


class TestTextToSpeech(unittest.TestCase):
    """
    TextToSpeech tests.
    """

    def testTextToSpeech(self):
        """
        Test generating speech for text
        """

        tts = TextToSpeech()

        # Check that data is generated
        self.assertGreater(len(tts("This is a test")), 0)

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

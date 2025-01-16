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

    def testESPnet(self):
        """
        Test generating speech for text with an ESPnet model
        """

        tts = TextToSpeech()

        # Check that data is generated
        speech, rate = tts("This is a test")

        self.assertGreater(len(speech), 0)
        self.assertEqual(rate, 22050)

    def testKokoro(self):
        """
        Test generating speech for text with a Kokoro model
        """

        tts = TextToSpeech("neuml/kokoro-int8-onnx", maxtokens=2)

        # Check that data is generated
        speech, rate = tts("This is a test")

        self.assertGreater(len(speech), 0)
        self.assertEqual(rate, 22050)

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

    def testSpeechT5(self):
        """
        Test generating speech for text with a SpeechT5 model
        """

        tts = TextToSpeech("neuml/txtai-speecht5-onnx")

        # Check that data is generated
        speech, rate = tts("This is a test")

        self.assertGreater(len(speech), 0)
        self.assertEqual(rate, 22050)

    def testStreaming(self):
        """
        Test streaming speech generation
        """

        tts = TextToSpeech()

        # Check that data is generated
        speech, rate = list(tts("This is a test. And another".split(), stream=True))[0]

        # Check that data is generated
        self.assertGreater(len(speech), 0)
        self.assertEqual(rate, 22050)

"""
TextToSpeech module tests
"""

import unittest

from unittest.mock import call, patch

import numpy as np

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

    @patch("txtai.pipeline.audio.texttospeech.Kokoro")
    @patch.object(TextToSpeech, "hasfile")
    def testStreamingOptions(self, hasfile, kokoro):
        """
        Test backend options reach both flushed and final streaming segments
        """

        hasfile.side_effect = lambda _, name: name in ("model.onnx", "voices.json")
        audio = np.array([0.1, -0.1], dtype=np.float32)
        kokoro.return_value.return_value = (audio, 24000)
        tts = TextToSpeech("neuml/kokoro-int8-onnx", rate=None)

        chunks = list(tts(iter(["One ", "two ", "three.", "Tail"]), stream=True, speaker="af", speed=1.5, transcribe=False))

        self.assertEqual(
            kokoro.return_value.call_args_list,
            [call("One two three.", "af", speed=1.5, transcribe=False), call("Tail", "af", speed=1.5, transcribe=False)],
        )
        self.assertEqual(len(chunks), 2)
        for data, rate in chunks:
            np.testing.assert_array_equal(data, audio)
            self.assertEqual(rate, 24000)

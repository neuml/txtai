"""
Microphone module tests
"""

import unittest

from unittest.mock import patch

import soundfile as sf

from txtai.pipeline import Microphone

# pylint: disable = C0411
from utils import Utils


class TestMicrophone(unittest.TestCase):
    """
    Microphone tests.
    """

    @patch("speech_recognition.Recognizer.listen")
    @patch("speech_recognition.Recognizer.adjust_for_ambient_noise")
    @patch("speech_recognition.Microphone")
    # pylint: disable=C0115,C0116,W0613
    def testMicrophone(self, microphone, ambient, listen):
        """
        Test listening to microphone
        """

        class Audio:
            def __init__(self):
                self.frame_data, self.sample_rate = None, None

            def get_wav_data(self):
                return self.frame_data

        def speech(device):
            # Read audio data
            raw, samplerate = sf.read(Utils.PATH + "/Make_huge_profits.wav")

            audio = Audio()
            audio.frame_data, audio.sample_rate = raw, samplerate
            return audio

        def nospeech(device):
            audio = Audio()
            audio.sample_rate = 16000
            audio.frame_data = b"\x00\x00" * int(audio.sample_rate * 30 / 1000)
            return audio

        microphone.return_value.__enter__.return_value = (0, 1)
        ambient.return_value = True

        pipeline = Microphone()

        listen.side_effect = speech
        self.assertIsNotNone(pipeline([1]))

        listen.side_effect = nospeech
        self.assertIsNone(pipeline.listen(microphone))

"""
AudioStream module tests
"""

import unittest

from unittest.mock import patch

import soundfile as sf

from txtai.pipeline import AudioStream

# pylint: disable = C0411
from utils import Utils


class TestAudioStream(unittest.TestCase):
    """
    AudioStream tests.
    """

    @patch("sounddevice.play")
    def testAudioStream(self, play):
        """
        Test playing audio
        """

        play.return_value = True

        # Read audio data
        raw, samplerate = sf.read(Utils.PATH + "/Make_huge_profits.wav")

        audio = AudioStream(samplerate)
        self.assertIsNotNone(audio([raw, AudioStream.COMPLETE]))

        # Wait for completion
        audio.wait()

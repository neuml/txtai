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
        audio, rate = sf.read(Utils.PATH + "/Make_huge_profits.wav")

        stream = AudioStream()
        self.assertIsNotNone(stream([(audio, rate), AudioStream.COMPLETE]))

        # Wait for completion
        stream.wait()

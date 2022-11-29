"""
Transcription module tests
"""

import unittest

import soundfile as sf

from txtai.pipeline import Transcription

# pylint: disable = C0411
from utils import Utils


class TestTranscription(unittest.TestCase):
    """
    Transcription tests.
    """

    def testArray(self):
        """
        Test audio data to text transcription
        """

        transcribe = Transcription()

        # Read audio data
        raw, samplerate = sf.read(Utils.PATH + "/Make_huge_profits.wav")

        self.assertEqual(transcribe(raw, samplerate), "Make huge profits without working make up to one hundred thousand dollars a day")

    def testFile(self):
        """
        Test audio file to text transcription
        """

        transcribe = Transcription()

        self.assertEqual(
            transcribe(Utils.PATH + "/Make_huge_profits.wav"), "Make huge profits without working make up to one hundred thousand dollars a day"
        )

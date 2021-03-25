"""
Transcription module tests
"""

import unittest

from txtai.pipeline import Transcription

# pylint: disable = C0411
from utils import Utils


class TestTranscription(unittest.TestCase):
    """
    Transcription tests
    """

    def testTranscription(self):
        """
        Test audio to text transcription
        """

        transcribe = Transcription()

        self.assertEqual(
            transcribe(Utils.PATH + "/Make_huge_profits.wav"), "Make huge profits without working make up to one hundred thousand dollars a day"
        )

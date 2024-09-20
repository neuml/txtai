"""
Transcription module tests
"""

import unittest

import numpy as np
import soundfile as sf

from scipy import signal

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

        self.assertEqual(transcribe((raw, samplerate)), "Make huge profits without working make up to one hundred thousand dollars a day")
        self.assertEqual(transcribe(raw, samplerate), "Make huge profits without working make up to one hundred thousand dollars a day")

    def testChunks(self):
        """
        Test splitting transcription into chunks
        """

        transcribe = Transcription()

        result = transcribe(Utils.PATH + "/Make_huge_profits.wav", join=False)[0]

        self.assertIsInstance(result["raw"], np.ndarray)
        self.assertIsNotNone(result["rate"])
        self.assertEqual(result["text"], "Make huge profits without working make up to one hundred thousand dollars a day")

    def testFile(self):
        """
        Test audio file to text transcription
        """

        transcribe = Transcription()

        self.assertEqual(
            transcribe(Utils.PATH + "/Make_huge_profits.wav"), "Make huge profits without working make up to one hundred thousand dollars a day"
        )

    def testGenerateArguments(self):
        """
        Test transcription with generation keyword arguments
        """

        transcribe = Transcription()

        # Read audio data
        raw, samplerate = sf.read(Utils.PATH + "/Make_huge_profits.wav")

        self.assertEqual(
            transcribe(raw, samplerate, language="English", task="transcribe"),
            "Make huge profits without working make up to one hundred thousand dollars a day",
        )

    def testResample(self):
        """
        Test resampled audio file to text transcription
        """

        transcribe = Transcription()

        # Read audio data
        raw, samplerate = sf.read(Utils.PATH + "/Make_huge_profits.wav")

        # Resample for testing
        samples = round(len(raw) * float(22050) / samplerate)
        raw, samplerate = signal.resample(raw, samples), 22050

        self.assertEqual(transcribe(raw, samplerate), "Make huge profits without working make up to one hundred thousand dollars a day")

    def testStereo(self):
        """
        Test audio file in stereo to text transcription
        """

        transcribe = Transcription()

        # Read audio data
        raw, samplerate = sf.read(Utils.PATH + "/Make_huge_profits.wav")

        # Convert mono to stereo
        raw = np.column_stack((raw, raw))

        self.assertEqual(transcribe(raw, samplerate), "Make huge profits without working make up to one hundred thousand dollars a day")

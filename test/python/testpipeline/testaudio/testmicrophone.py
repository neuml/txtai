"""
Microphone module tests
"""

import unittest

from unittest.mock import patch

import numpy as np
import soundfile as sf

from txtai.pipeline import Microphone

# pylint: disable = C0411
from utils import Utils


class TestMicrophone(unittest.TestCase):
    """
    Microphone tests.
    """

    # pylint: disable=C0115,C0116
    @patch("sounddevice.RawInputStream")
    def testMicrophone(self, inputstream):
        """
        Test listening to microphone
        """

        class RawInputStream:
            def __init__(self, **kwargs):
                self.args = kwargs

                # Read audio data
                self.index, self.passes = 0, 0
                audio, self.samplerate = sf.read(Utils.PATH + "/Make_huge_profits.wav")

                # Convert data to PCM
                self.audio = self.int16(audio)

                # Start with random data to test that speech is not detected
                self.data = np.concatenate((self.audio * 50, np.zeros(shape=self.audio.shape, dtype=np.int16)))

            def start(self):
                pass

            def stop(self):
                pass

            def read(self, size):
                # Get chunk
                chunk = self.data[self.index : self.index + size]
                self.index += size

                # Initial pass is random data, 2nd pass is speech data
                if self.index > len(self.data):
                    if not self.passes:
                        self.index, self.passes = 0, self.passes + 1
                        self.data = self.audio
                    elif self.index >= len(self.audio) * 10:
                        # Break out of loop if speech continues to not be detected
                        raise IOError("Data exhausted")

                return chunk, False

            def int16(self, data):
                i = np.iinfo(np.int16)
                absmax = 2 ** (i.bits - 1)
                offset = i.min + absmax
                return (data * absmax + offset).clip(i.min, i.max).astype(np.int16)

        # Mock input stream
        inputstream.side_effect = RawInputStream

        # Create microphone pipeline and read data
        pipeline = Microphone()
        data, rate = pipeline()

        # Validate sample rate and length of data
        self.assertEqual(len(data), 91220)
        self.assertEqual(rate, 16000)

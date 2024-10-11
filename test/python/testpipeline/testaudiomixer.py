"""
AudioMixer module tests
"""

import unittest

import numpy as np

from txtai.pipeline import AudioMixer


class TestAudioStream(unittest.TestCase):
    """
    AudioStream tests.
    """

    def testAudioStream(self):
        """
        Test mixing audio streams
        """

        audio1 = np.random.rand(2, 5000), 100
        audio2 = np.random.rand(2, 5000), 100

        mixer = AudioMixer()
        audio, rate = mixer((audio1, audio2))

        self.assertEqual(audio.shape, (2, 5000))
        self.assertEqual(rate, 100)

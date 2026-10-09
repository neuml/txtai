"""
Signal module tests
"""

import unittest

import numpy as np

from txtai.pipeline.audio.signal import Signal


class TestSignal(unittest.TestCase):
    """
    Signal tests.
    """

    def testTrimPartialWindow(self):
        """
        Test that short inputs and partial windows are not lost when trimming
        """

        # At 1000 Hz the detection window has 40 samples.
        pattern = np.array([1.0, -1.0], dtype=np.float32)
        for size in (0, 20, 43, 83):
            audio = np.resize(pattern, size)
            for trailing in (False, True):
                with self.subTest(size=size, trailing=trailing):
                    np.testing.assert_array_equal(Signal.trim(audio, 1000, trailing=trailing), audio)

        # Leading silence is still removed while the partial tail is retained.
        audio = np.resize(pattern, 83)
        padded = np.concatenate((np.zeros(40, dtype=np.float32), audio))
        np.testing.assert_array_equal(Signal.trim(padded, 1000, trailing=False), audio)

"""
AudioStream module tests
"""

import tempfile
import unittest

from pathlib import Path

from unittest.mock import patch

import numpy as np
import soundfile as sf

from txtai.pipeline import AudioStream

# pylint: disable=C0411
from utils import Utils


class TestAudioStream(unittest.TestCase):
    """
    AudioStream tests.
    """

    @patch("sounddevice.play")
    def testStereoResampling(self, play):
        """
        Test resampling preserves stereo channels and adjusts frame count
        """

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, np.tile([0.25, -0.5], (640, 1)), 16000)
            audio, rate = sf.read(path)

        for target, frames in ((8000, 320), (32000, 1280)):
            with self.subTest(rate=target):
                stream = AudioStream(rate=target)
                stream([(audio, rate), AudioStream.COMPLETE])
                stream.wait()

                played, outputrate = play.call_args.args
                self.assertEqual(outputrate, target)
                self.assertEqual(played.shape, (frames, 2))
                np.testing.assert_allclose(played, np.tile([0.25, -0.5], (frames, 1)))

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

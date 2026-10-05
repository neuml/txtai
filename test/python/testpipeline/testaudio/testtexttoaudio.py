"""
TextToAudio module tests
"""

import unittest

from txtai.pipeline import TextToAudio


class TestTextToAudio(unittest.TestCase):
    """
    TextToAudio tests.
    """

    def testTextToAudio(self):
        """
        Test generating audio for text
        """

        tta = TextToAudio("hf-internal-testing/tiny-random-MusicgenForConditionalGeneration")

        # Check that data is generated
        audio, rate = tta("This is a test")

        self.assertGreater(len(audio), 0)
        self.assertEqual(rate, 24000)

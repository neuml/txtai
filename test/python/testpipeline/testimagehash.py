"""
ImageHash module tests
"""

import unittest

from PIL import Image

from txtai.pipeline import ImageHash

# pylint: disable = C0411
from utils import Utils


class TestImageHash(unittest.TestCase):
    """
    ImageHash tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Caches an image to hash
        """

        cls.image = Image.open(Utils.PATH + "/books.jpg")

    def testArray(self):
        """
        Test numpy return type
        """

        ihash = ImageHash(strings=False)
        self.assertEqual(ihash(self.image).shape, (64,))

    def testAverage(self):
        """
        Test average hash
        """

        ihash = ImageHash("average")
        self.assertIn(ihash(self.image), ["0859dd04bfbfbf00", "0859dd04ffbfbf00"])

    def testColor(self):
        """
        Test color hash
        """

        ihash = ImageHash("color")
        self.assertIn(ihash(self.image), ["1ffffe02000e000c0e0000070000", "1ff8fe03000e00070e0000070000"])

    def testDifference(self):
        """
        Test difference hash
        """

        ihash = ImageHash("difference")
        self.assertEqual(ihash(self.image), "d291996d6969686a")

    def testPerceptual(self):
        """
        Test perceptual hash
        """

        ihash = ImageHash("perceptual")
        self.assertEqual(ihash(self.image), "8be8418577b331b9")

    def testWavelet(self):
        """
        Test wavelet hash
        """

        ihash = ImageHash("wavelet")
        self.assertEqual(ihash(Utils.PATH + "/books.jpg"), "68015d85bfbf3f00")

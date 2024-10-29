"""
Objects module tests
"""

import unittest

from txtai.pipeline import Objects

# pylint: disable = C0411
from utils import Utils


class TestObjects(unittest.TestCase):
    """
    Object detection tests.
    """

    def testClassification(self):
        """
        Test object detection using an image classification model
        """

        objects = Objects(classification=True, threshold=0.3)
        self.assertEqual(objects(Utils.PATH + "/books.jpg")[0][0], "library")

    def testDetection(self):
        """
        Test object detection using an object detection model
        """

        objects = Objects()
        self.assertEqual(objects(Utils.PATH + "/books.jpg")[0][0], "book")

    def testFlatten(self):
        """
        Test object detection using an object detection model, flatten to return only objects
        """

        objects = Objects()
        self.assertEqual(objects(Utils.PATH + "/books.jpg", flatten=True)[0], "book")

"""
Models module tests
"""

import unittest

from unittest.mock import patch

import torch

from txtai.models import Models


class TestModels(unittest.TestCase):
    """
    Models tests.
    """

    @patch("torch.cuda.is_available")
    def testDeviceid(self, cuda):
        """
        Test the deviceid method
        """

        cuda.return_value = True
        self.assertEqual(Models.deviceid(True), 0)
        self.assertEqual(Models.deviceid(False), -1)
        self.assertEqual(Models.deviceid(0), 0)
        self.assertEqual(Models.deviceid(1), 1)

        # Test direct torch device
        # pylint: disable=E1101
        self.assertEqual(Models.deviceid(torch.device("cpu")), torch.device("cpu"))

        cuda.return_value = False
        self.assertEqual(Models.deviceid(True), -1)
        self.assertEqual(Models.deviceid(False), -1)
        self.assertEqual(Models.deviceid(0), -1)
        self.assertEqual(Models.deviceid(1), -1)

    def testDevice(self):
        """
        Test the device method
        """

        # pylint: disable=E1101
        self.assertEqual(Models.device("cpu"), torch.device("cpu"))
        self.assertEqual(Models.device(torch.device("cpu")), torch.device("cpu"))

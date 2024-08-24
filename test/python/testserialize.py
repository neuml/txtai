"""
Serialize module tests
"""

import os
import unittest

from unittest.mock import patch

from txtai.serialize import Serialize, SerializeFactory


class TestSerialize(unittest.TestCase):
    """
    Serialize tests.
    """

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        serialize = Serialize()

        self.assertRaises(NotImplementedError, serialize.loadstream, None)
        self.assertRaises(NotImplementedError, serialize.savestream, None, None)
        self.assertRaises(NotImplementedError, serialize.loadbytes, None)
        self.assertRaises(NotImplementedError, serialize.savebytes, None)

    def testMessagePack(self):
        """
        Test MessagePack encoder
        """

        serializer = SerializeFactory.create()
        self.assertEqual(serializer.loadbytes(serializer.savebytes("test")), "test")

    @patch.dict(os.environ, {"ALLOW_PICKLE": "False"})
    def testPickleDisabled(self):
        """
        Test disabled pickle serialization
        """

        # Validate an error is raised
        with self.assertRaises(ValueError):
            serializer = SerializeFactory.create("pickle")
            data = serializer.savebytes("Test")
            serializer.loadbytes(data)

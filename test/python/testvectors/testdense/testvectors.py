"""
Vectors module tests
"""

import json
import os
import tempfile
import unittest

import numpy as np

from txtai.vectors import Vectors, Recovery


class TestVectors(unittest.TestCase):
    """
    Vectors tests.
    """

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        vectors = Vectors(None, None, None)

        self.assertRaises(NotImplementedError, vectors.load, None)
        self.assertRaises(NotImplementedError, vectors.encode, None)

    def testNormalize(self):
        """
        Test batch normalize and single input normalize are equal
        """

        vectors = Vectors(None, None, None)

        # Generate data
        data1 = np.random.rand(5, 5).astype(np.float32)
        data2 = data1.copy()

        # Keep original data to ensure it changed
        original = data1.copy()

        # Normalize data
        vectors.normalize(data1)
        for x in data2:
            vectors.normalize(x)

        # Test both data arrays are the same and changed from original
        self.assertTrue(np.allclose(data1, data2))
        self.assertFalse(np.allclose(data1, original))

    def testRecovery(self):
        """
        Test vectors recovery failure
        """

        # Checkpoint directory
        checkpoint = os.path.join(tempfile.gettempdir(), "recovery")
        os.makedirs(checkpoint, exist_ok=True)

        # Create empty file
        # pylint: disable=R1732
        f = open(os.path.join(checkpoint, "id"), "w", encoding="utf-8")
        f.close()

        # Create the recovery instance with an empty checkpoint file
        recovery = Recovery(checkpoint, "id", np.load)
        self.assertIsNone(recovery(["a"], ["hash-a"]))

    def testRecoveryLegacyCheckpoint(self):
        """
        Test that a checkpoint written before ids tracking existed (no companion .ids file) is
        treated as unavailable rather than trusted
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "recovery-legacy")
        os.makedirs(checkpoint, exist_ok=True)

        embeddings = np.random.rand(2, 4).astype(np.float32)
        with open(os.path.join(checkpoint, "id"), "wb") as f:
            np.save(f, embeddings)

        # No "id.ids" companion file present
        recovery = Recovery(checkpoint, "id", np.load)
        self.assertIsNone(recovery(["a", "b"], ["hash-a", "hash-b"]))

    def testRecoveryMismatch(self):
        """
        Test that recovery rejects a batch whose checkpointed ids don't match the current run's ids
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "recovery-mismatch")
        os.makedirs(checkpoint, exist_ok=True)

        embeddings = np.random.rand(2, 4).astype(np.float32)
        with open(os.path.join(checkpoint, "id"), "wb") as f:
            np.save(f, embeddings)

        with open(os.path.join(checkpoint, "id.ids"), "w", encoding="utf-8") as f:
            f.write(json.dumps([["a", "b"], ["hash-a", "hash-b"]]) + "\n")

        recovery = Recovery(checkpoint, "id", np.load)
        self.assertIsNone(recovery(["a", "c"], ["hash-a", "hash-c"]))

    def testRecoveryContentMismatch(self):
        """
        Test that recovery rejects a batch whose ids match but whose content checksums don't - the
        same id can carry different content across runs (an updated row from the source, for
        example), and matching ids alone must not be trusted in that case
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "recovery-content-mismatch")
        os.makedirs(checkpoint, exist_ok=True)

        embeddings = np.random.rand(2, 4).astype(np.float32)
        with open(os.path.join(checkpoint, "id"), "wb") as f:
            np.save(f, embeddings)

        with open(os.path.join(checkpoint, "id.ids"), "w", encoding="utf-8") as f:
            f.write(json.dumps([["a", "b"], ["hash-a", "hash-b"]]) + "\n")

        # Same ids as the checkpoint, but "b"'s content checksum has changed
        recovery = Recovery(checkpoint, "id", np.load)
        self.assertIsNone(recovery(["a", "b"], ["hash-a", "hash-b-changed"]))

    def testRecoveryMatch(self):
        """
        Test that recovery returns the checkpointed embeddings when their recorded ids and content
        checksums match the current run's ids and checksums
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "recovery-match")
        os.makedirs(checkpoint, exist_ok=True)

        embeddings = np.random.rand(2, 4).astype(np.float32)
        with open(os.path.join(checkpoint, "id"), "wb") as f:
            np.save(f, embeddings)

        with open(os.path.join(checkpoint, "id.ids"), "w", encoding="utf-8") as f:
            f.write(json.dumps([["a", "b"], ["hash-a", "hash-b"]]) + "\n")

        recovery = Recovery(checkpoint, "id", np.load)
        result = recovery(["a", "b"], ["hash-a", "hash-b"])
        self.assertTrue(np.allclose(result, embeddings))

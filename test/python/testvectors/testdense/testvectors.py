"""
Vectors module tests
"""

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

    def write(self, path, checksums=None, embeddings=None):
        """
        Writes a checkpoint and an optional companion checksums file.

        Args:
            path: checkpoint file to write
            checksums: list of checksums to write to a companion file, or None to omit it (simulating
                       a checkpoint written before checksums existed)
            embeddings: embeddings array to write, or None to write an empty checkpoint file
        """

        with open(path, "wb") as f:
            if embeddings is not None:
                np.save(f, embeddings, allow_pickle=False)

        if checksums is not None:
            with open(f"{path}.checksums", "w", encoding="utf-8") as f:
                for checksum in checksums:
                    f.write(checksum + "\n")

    def testRecovery(self):
        """
        Test vectors recovery failure
        """

        # Checkpoint directory
        checkpoint = os.path.join(tempfile.gettempdir(), "recovery")
        os.makedirs(checkpoint, exist_ok=True)

        # Create empty file
        self.write(os.path.join(checkpoint, "id"))

        # Create the recovery instance with an empty checkpoint file
        recovery = Recovery(checkpoint, "id", np.load)
        self.assertIsNone(recovery("checksum-a"))

    def testRecoveryLegacyCheckpoint(self):
        """
        Test that a checkpoint written before checksum tracking existed (no companion checksums
        file) is still read back and trusted as-is
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "recovery-legacy")
        os.makedirs(checkpoint, exist_ok=True)

        embeddings = np.random.rand(2, 4).astype(np.float32)
        self.write(os.path.join(checkpoint, "id"), embeddings=embeddings)

        # No companion checksums file present
        recovery = Recovery(checkpoint, "id", np.load)
        result = recovery("checksum-a")
        self.assertTrue(np.allclose(result, embeddings))

    def testRecoveryMismatch(self):
        """
        Test that recovery rejects a batch whose checkpointed checksum doesn't match the current
        run's checksum for that batch
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "recovery-mismatch")
        os.makedirs(checkpoint, exist_ok=True)

        embeddings = np.random.rand(2, 4).astype(np.float32)
        self.write(os.path.join(checkpoint, "id"), checksums=["checksum-a"], embeddings=embeddings)

        recovery = Recovery(checkpoint, "id", np.load)
        self.assertIsNone(recovery("checksum-b"))

    def testRecoveryContentMismatch(self):
        """
        Test that recovery rejects a batch whose ids match but whose content checksum doesn't - the
        same id can carry different content across runs (an updated row from the source, for
        example), and matching ids alone must not be trusted in that case. The checksum covers both
        ids and content, so this looks identical to any other checksum mismatch from here
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "recovery-content-mismatch")
        os.makedirs(checkpoint, exist_ok=True)

        embeddings = np.random.rand(2, 4).astype(np.float32)
        self.write(os.path.join(checkpoint, "id"), checksums=["checksum-ab"], embeddings=embeddings)

        # Same ids as the checkpoint, but content changed means the checksum won't match
        recovery = Recovery(checkpoint, "id", np.load)
        self.assertIsNone(recovery("checksum-ab-changed"))

    def testRecoveryMatch(self):
        """
        Test that recovery returns the checkpointed embeddings when their recorded checksum matches
        the current run's checksum
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "recovery-match")
        os.makedirs(checkpoint, exist_ok=True)

        embeddings = np.random.rand(2, 4).astype(np.float32)
        self.write(os.path.join(checkpoint, "id"), checksums=["checksum-ab"], embeddings=embeddings)

        recovery = Recovery(checkpoint, "id", np.load)
        result = recovery("checksum-ab")
        self.assertTrue(np.allclose(result, embeddings))

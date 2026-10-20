"""
Recovery module
"""

import os
import shutil


class Recovery:
    """
    Vector embeddings recovery. This class handles streaming embeddings from a vector checkpoint file.

    Each batch in the checkpoint file is preceded by a checksum line covering that batch's ids and raw
    content. Before trusting a recovered batch, this class confirms the recorded checksum still matches
    the checksum of the current run's batch at that position. Matching ids alone isn't sufficient - the
    same id can carry different content across runs (an updated row from the source, for example) - so
    the checksum folds in content as well as ids. If the document stream was reordered, changed, or its
    content edited since the checkpoint was written, a stale batch would otherwise be paired with the
    wrong documents. A checkpoint written before this checksum line existed has no marker to confirm
    that, so it's treated as unavailable rather than trusted.
    """

    # Marks a checkpoint file as carrying a per-batch checksum line before each batch's embeddings.
    # Lets a checkpoint written before checksums existed be told apart and treated as unavailable.
    MAGIC = "TXTAI-CHECKSUMS-V1\n"

    def __init__(self, checkpoint, vectorsid, load):
        """
        Creates a Recovery instance.

        Args:
            checkpoint: checkpoint directory
            vectorsid: vectors uid for current configuration
            load: load embeddings method
        """

        self.spool, self.path, self.load = None, None, load

        # Get unique file id
        path = f"{checkpoint}/{vectorsid}"
        if os.path.exists(path):
            # Generate recovery path
            self.path = f"{checkpoint}/recovery"

            # Copy current checkpoint to recovery
            shutil.copyfile(path, self.path)

            # Open file and return
            # pylint: disable=R1732
            spool = open(self.path, "rb")

            # Only trust files carrying the checksum format marker - a checkpoint written before
            # this existed has no way to confirm its batches still match
            if spool.readline().decode("utf-8", errors="replace") == self.MAGIC:
                self.spool = spool
            else:
                spool.close()
                os.remove(self.path)
                self.path = None

    def __call__(self, checksum):
        """
        Reads and returns the next batch of embeddings, provided the checkpoint's recorded checksum
        for that batch still matches the batch about to be built.

        Args:
            checksum: checksum for the batch about to be built, used to confirm the recovered batch's
                      ids and content weren't changed since the checkpoint was written

        Returns
            batch of embeddings, or None if unavailable or the checkpoint no longer lines up
        """

        if not self.spool:
            return None

        try:
            line = self.spool.readline()
            if not line:
                raise EOFError

            recoveredchecksum = line.decode("utf-8").strip()
            embeddings = self.load(self.spool)
        except EOFError:
            # End of spool file, cleanup
            self.cleanup()
            return None

        if recoveredchecksum != checksum:
            # The document stream no longer lines up with the checkpoint, either a different document
            # (id) or the same id with changed content (checksum). Every later batch position is
            # unreliable too once this happens, so stop trusting the recovery file for the rest of this run.
            self.cleanup()
            return None

        return embeddings

    def cleanup(self):
        """
        Closes and removes the recovery spool file.
        """

        if self.spool:
            self.spool.close()
            os.remove(self.path)

        self.spool, self.path = None, None

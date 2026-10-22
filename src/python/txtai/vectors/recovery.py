"""
Recovery module
"""

import os
import shutil


class Recovery:
    """
    Vector embeddings recovery. This class handles streaming embeddings from a vector checkpoint file.

    A checkpoint's embeddings are stored in one file, unchanged from the original checkpoint format. A
    separate, optional companion file holds one checksum per batch, covering that batch's ids and raw
    content. When present, a recovered batch is only trusted once its recorded checksum matches the
    checksum of the current run's batch at that position - matching ids alone isn't sufficient, since
    the same id can carry different content across runs (an updated row from the source, for example).
    A checkpoint written before this checksum file existed has no such file to find, so its batches are
    read back and trusted as-is, same as before checksums existed.
    """

    def __init__(self, checkpoint, vectorsid, load):
        """
        Creates a Recovery instance.

        Args:
            checkpoint: checkpoint directory
            vectorsid: vectors uid for current configuration
            load: load embeddings method
        """

        self.spool, self.path = None, None
        self.checksums, self.checksumspath = None, None
        self.load = load

        # Get unique file id
        path = f"{checkpoint}/{vectorsid}"
        if os.path.exists(path):
            # Generate recovery path
            self.path = f"{checkpoint}/recovery"

            # Copy current checkpoint to recovery
            shutil.copyfile(path, self.path)

            # Open file and return
            # pylint: disable=R1732
            self.spool = open(self.path, "rb")

            # Checksums are optional - a checkpoint from before this file existed has none, so its
            # batches are trusted without verification
            checksumspath = f"{checkpoint}/{vectorsid}.checksums"
            if os.path.exists(checksumspath):
                self.checksumspath = f"{checkpoint}/recovery.checksums"
                shutil.copyfile(checksumspath, self.checksumspath)

                # pylint: disable=R1732
                self.checksums = open(self.checksumspath, encoding="utf-8")

    def __call__(self, checksum):
        """
        Reads and returns the next batch of embeddings. When a checksums file is present, the batch is
        only returned once its recorded checksum matches the checksum of the batch about to be built.

        Args:
            checksum: checksum for the batch about to be built, used to confirm the recovered batch's
                      ids and content weren't changed since the checkpoint was written

        Returns
            batch of embeddings, or None if unavailable or the checkpoint no longer lines up
        """

        if not self.spool:
            return None

        try:
            if self.checksums:
                line = self.checksums.readline()
                if not line:
                    raise EOFError

                if line.strip() != checksum:
                    # The document stream no longer lines up with the checkpoint, either a different
                    # document (id) or the same id with changed content (checksum). Every later batch
                    # position is unreliable too once this happens, so stop trusting the recovery file
                    # for the rest of this run.
                    self.cleanup()
                    return None

            return self.load(self.spool)
        except EOFError:
            # End of spool file, cleanup
            self.cleanup()
            return None

    def cleanup(self):
        """
        Closes and removes the recovery spool and checksums files.
        """

        if self.spool:
            self.spool.close()
            os.remove(self.path)

        if self.checksums:
            self.checksums.close()
            os.remove(self.checksumspath)

        self.spool, self.path = None, None
        self.checksums, self.checksumspath = None, None

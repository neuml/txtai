"""
Recovery module
"""

import json
import os
import shutil


class Recovery:
    """
    Vector embeddings recovery. This class handles streaming embeddings from a vector checkpoint file.

    Alongside the embeddings, a checkpoint carries a companion ids file with one JSON-encoded
    [ids, checksums] pair per batch. Before trusting a recovered batch, this class confirms both the
    recorded ids and the recorded content checksums still match the current run's documents at that
    position. Matching ids alone isn't sufficient - the same id can carry different content across runs
    (an updated row from the source, for example) - so each document's raw content is also hashed and
    compared. If the document stream was reordered, changed, or its content edited since the checkpoint
    was written, a stale batch would otherwise be paired with the wrong documents. A checkpoint written
    before this ids file existed has no way to be verified, so it's treated as unavailable rather than
    trusted.
    """

    def __init__(self, checkpoint, vectorsid, load):
        """
        Creates a Recovery instance.

        Args:
            checkpoint: checkpoint directory
            vectorsid: vectors uid for current configuration
            load: load embeddings method
        """

        self.spool, self.idspool, self.path, self.idspath, self.load = None, None, None, None, load

        # Get unique file id
        path = f"{checkpoint}/{vectorsid}"
        idspath = f"{path}.ids"
        if os.path.exists(path) and os.path.exists(idspath):
            # Generate recovery paths
            self.path = f"{checkpoint}/recovery"
            self.idspath = f"{checkpoint}/recovery.ids"

            # Copy current checkpoint to recovery
            shutil.copyfile(path, self.path)
            shutil.copyfile(idspath, self.idspath)

            # Open files an return
            # pylint: disable=R1732
            self.spool = open(self.path, "rb")
            # pylint: disable=R1732
            self.idspool = open(self.idspath, "r", encoding="utf-8")

    def __call__(self, ids, hashes):
        """
        Reads and returns the next batch of embeddings, provided the checkpoint's recorded ids and
        content checksums for that batch still match the batch about to be built.

        Args:
            ids: ids for the batch about to be built, used to confirm a recovered batch applies to it
            hashes: content checksums for the batch about to be built, used to confirm the recovered
                    batch's documents weren't changed since the checkpoint was written

        Returns
            batch of embeddings, or None if unavailable or the checkpoint no longer lines up
        """

        if not self.spool:
            return None

        try:
            line = self.idspool.readline()
            if not line:
                raise EOFError

            recoveredids, recoveredhashes = json.loads(line)
            embeddings = self.load(self.spool)
        except EOFError:
            # End of spool file, cleanup
            self.cleanup()
            return None

        if recoveredids != ids or recoveredhashes != hashes:
            # The document stream no longer lines up with the checkpoint, either a different document
            # (id) or the same id with changed content (checksum). Every later batch position is
            # unreliable too once this happens, so stop trusting the recovery file for the rest of this run.
            self.cleanup()
            return None

        return embeddings

    def cleanup(self):
        """
        Closes and removes the recovery spool files.
        """

        if self.spool:
            self.spool.close()
            os.remove(self.path)

        if self.idspool:
            self.idspool.close()
            os.remove(self.idspath)

        self.spool, self.idspool, self.path, self.idspath = None, None, None, None

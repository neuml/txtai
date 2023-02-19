"""
Tar module
"""

import os
import tarfile

from .compress import Compress


class Tar(Compress):
    """
    Tar compression
    """

    def pack(self, path, output):
        # Infer compression type
        compression = self.compression(output)

        with tarfile.open(output, f"w:{compression}" if compression else "w") as tar:
            tar.add(path, arcname=".")

    def unpack(self, path, output):
        # Infer compression type
        compression = self.compression(path)

        with tarfile.open(path, f"r:{compression}" if compression else "r") as tar:
            # Validate paths
            for member in tar.getmembers():
                fullpath = os.path.join(path, member.name)
                if not self.validate(path, fullpath):
                    raise IOError(f"Invalid tar entry: {member.name}")

            tar.extractall(output)

    def compression(self, path):
        """
        Gets compression type for path.

        Args:
            path: path to file

        Returns:
            compression type
        """

        # Infer compression type from last path component. Limit to supported types.
        compression = path.lower().split(".")[-1]
        return compression if compression in ("bz2", "gz", "xz") else None

"""
Zip module
"""

import os

from zipfile import ZipFile, ZIP_DEFLATED

from .base import Compress


class Zip(Compress):
    """
    Zip compression
    """

    def pack(self, path, output):
        with ZipFile(output, "w", ZIP_DEFLATED) as zfile:
            for root, _, files in sorted(os.walk(path)):
                for f in files:
                    zfile.write(os.path.join(root, f), arcname=f)

    def unpack(self, path, output):
        with ZipFile(path, "r") as zfile:
            # Validate path if directory specified
            for fullpath in zfile.namelist():
                if os.path.dirname(fullpath) and not self.validate(path, fullpath):
                    raise IOError(f"Invalid zip entry: {fullpath}")

            zfile.extractall(output)

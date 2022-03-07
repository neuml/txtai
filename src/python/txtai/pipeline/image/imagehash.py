"""
ImageHash module
"""

import numpy as np

# Conditional import
try:
    from PIL import Image
    import imagehash

    PIL = True
except ImportError:
    PIL = False

from ..base import Pipeline


class ImageHash(Pipeline):
    """
    Generates perceptual image hashes. These hashes can be used to detect near-duplicate images. This method is not
    backed by machine learning models and not intended to find conceptually similar images.
    """

    def __init__(self, algorithm="average", size=8, strings=True):
        """
        Creates an ImageHash pipeline.

        Args:
            algorithm: image hashing algorithm (average, perceptual, difference, wavelet, color)
            size: hash size
            strings: outputs hex strings if True (default), otherwise the pipeline returns numpy arrays
        """

        if not PIL:
            raise ImportError('ImageHash pipeline is not available - install "pipeline" extra to enable')

        self.algorithm = algorithm
        self.size = size
        self.strings = strings

    def __call__(self, images):
        """
        Generates perceptual image hashes.

        Args:
            images: image|list

        Returns:
            list of hashes
        """

        # Convert single element to list
        values = [images] if not isinstance(images, list) else images

        # Open images if file strings
        values = [Image.open(image) if isinstance(image, str) else image for image in values]

        # Convert images to hashes
        hashes = [self.ihash(image) for image in values]

        # Return single element if single element passed in
        return hashes[0] if not isinstance(images, list) else hashes

    def ihash(self, image):
        """
        Gets an image hash for image.

        Args:
            image: PIL image

        Returns:
            hash as hex string
        """

        # Apply hash algorithm
        if self.algorithm == "perceptual":
            data = imagehash.phash(image, self.size)
        elif self.algorithm == "difference":
            data = imagehash.dhash(image, self.size)
        elif self.algorithm == "wavelet":
            data = imagehash.whash(image, self.size)
        elif self.algorithm == "color":
            data = imagehash.colorhash(image, self.size)
        else:
            # Default to average hash
            data = imagehash.average_hash(image, self.size)

        # Convert to output data type
        return str(data) if self.strings else data.hash.astype(np.float32).reshape(-1)

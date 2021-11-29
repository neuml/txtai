"""
Objects module
"""

# Conditional import
try:
    from PIL import Image

    PIL = True
except ImportError:
    PIL = False

from ..hfpipeline import HFPipeline


class Objects(HFPipeline):
    """
    Applies object detection models to images. Supports both object detection models and image classification models.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, classification=False, threshold=0.9):
        if not PIL:
            raise ImportError('Objects pipeline is not available - install "pipeline" extra to enable')

        super().__init__("image-classification" if classification else "object-detection", path, quantize, gpu, model)

        self.classification = classification
        self.threshold = threshold

    def __call__(self, images, flatten=False, workers=0):
        """
        Applies object detection/image classification models to images. Returns a list of (label, score).

        This method supports a single image or a list of images. If the input is an image, the return
        type is a 1D list of (label, score). If text is a list, a 2D list of (label, score) is
        returned with a row per image.

        Args:
            images: image|list
            flatten: flatten output to a list of objects
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            list of (label, score)
        """

        # Convert single element to list
        values = [images] if not isinstance(images, list) else images

        # Open images if file strings
        values = [Image.open(image) if isinstance(image, str) else image for image in values]

        # Run pipeline
        results = (
            self.pipeline(values, num_workers=workers)
            if self.classification
            else self.pipeline(values, threshold=self.threshold, num_workers=workers)
        )

        # Build list of (id, score)
        outputs = []
        for result in results:
            # Convert to (label, score) tuples
            result = [(x["label"], x["score"]) for x in result if x["score"] > self.threshold]

            # Sort by score descending
            result = sorted(result, key=lambda x: x[1], reverse=True)

            # Deduplicate labels
            unique = set()
            elements = []
            for label, score in result:
                if label not in unique:
                    elements.append(label if flatten else (label, score))
                    unique.add(label)

            outputs.append(elements)

        # Return single element if single element passed in
        return outputs[0] if not isinstance(images, list) else outputs

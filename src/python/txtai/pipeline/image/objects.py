"""
Objects module
"""

from ..hfpipeline import HFPipeline


class Objects(HFPipeline):
    """
    Applies object detection models to images. Supports both object detection models and image classification models.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, classification=False, threshold=0.9):
        super().__init__("image-classification" if classification else "object-detection", path, quantize, gpu, model)

        self.classification = classification
        self.threshold = threshold

    def __call__(self, images):
        """
        Applies object detection/image classification models to images. Returns a list of (label, score).

        This method supports a single image or a list of images. If the input is an image, the return
        type is a 1D list of (label, score). If text is a list, a 2D list of (label, score) is
        returned with a row per image.

        Args:
            images: image|list

        Returns:
            list of (label, score)
        """

        # Convert single element to list
        values = [images] if not isinstance(images, list) else images

        # Run pipeline
        results = self.pipeline(values) if self.classification else self.pipeline(values, threshold=self.threshold)

        # Build list of (id, score)
        scores = []
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
                    elements.append((label, score))
                    unique.add(label)

            scores.append(elements)

        # Return single element if single element passed in
        return scores[0] if not isinstance(images, list) else scores

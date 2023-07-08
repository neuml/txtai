"""
Caption module
"""

# Conditional import
try:
    from PIL import Image

    PIL = True
except ImportError:
    PIL = False

from ..hfpipeline import HFPipeline


class Caption(HFPipeline):
    """
    Constructs captions for images.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        if not PIL:
            raise ImportError('Captions pipeline is not available - install "pipeline" extra to enable')

        # Call parent constructor
        super().__init__("image-to-text", path, quantize, gpu, model, **kwargs)

    def __call__(self, images):
        """
        Builds captions for images.

        This method supports a single image or a list of images. If the input is an image, the return
        type is a string. If text is a list, a list of strings is returned

        Args:
            images: image|list

        Returns:
            list of captions
        """

        # Convert single element to list
        values = [images] if not isinstance(images, list) else images

        # Open images if file strings
        values = [Image.open(image) if isinstance(image, str) else image for image in values]

        # Get and clean captions
        captions = []
        for result in self.pipeline(values):
            text = " ".join([r["generated_text"] for r in result]).strip()
            captions.append(text)

        # Return single element if single element passed in
        return captions[0] if not isinstance(images, list) else captions

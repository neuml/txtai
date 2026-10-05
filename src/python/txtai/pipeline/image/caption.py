"""
Caption module
"""

# Conditional import
try:
    from PIL import Image

    PIL = True
except ImportError:
    PIL = False

from ..hfmodel import HFModel

# Core library imports
from ...util import Library

transformers = Library().transformers()


class Caption(HFModel):
    """
    Constructs captions for images.
    """

    def __init__(self, path=None, quantize=False, gpu=True, batch=64, **kwargs):
        if not PIL:
            raise ImportError('Captions pipeline is not available - install "pipeline" extra to enable')

        # Default model
        path = path if path else "ydshieh/vit-gpt2-coco-en"

        # Call parent constructor
        super().__init__(path, quantize, gpu, batch)

        # Captioning model
        if isinstance(path, tuple):
            self.model, self.tokenizer, self.processor = path
        else:
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(path, **kwargs)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
            self.processor = transformers.AutoImageProcessor.from_pretrained(path)

        # Move model to device
        self.model = self.model.to(self.device)

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
        for image in values:
            # Extract pixels
            pixels = self.processor(images=image, return_tensors="pt").to(self.device).pixel_values

            # Generate the caption
            outputs = self.model.generate(pixel_values=pixels, max_new_tokens=256)
            caption = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            captions.append(caption)

        # Return single element if single element passed in
        return captions[0] if not isinstance(images, list) else captions

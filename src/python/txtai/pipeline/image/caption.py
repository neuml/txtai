"""
Caption module
"""

# Conditional import
try:
    from PIL import Image

    PIL = True
except ImportError:
    PIL = False

import torch

from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor

from ..hfmodel import HFModel


class Caption(HFModel):
    """
    Constructs captions for images.
    """

    def __init__(self, path="ydshieh/vit-gpt2-coco-en", quantize=False, gpu=True, batch=64):
        """
        Constructs a new caption pipeline.

        Args:
            path: optional path to model, accepts Hugging Face model hub id or local path,
                  uses default model for task if not provided
            quantize: if model should be quantized, defaults to False
            gpu: True/False if GPU should be enabled, also supports a GPU device id
            batch: batch size used to incrementally process content
        """

        if not PIL:
            raise ImportError('Captions pipeline is not available - install "pipeline" extra to enable')

        # Call parent constructor
        super().__init__(path, quantize, gpu, batch)

        # load model and processor
        self.model = VisionEncoderDecoderModel.from_pretrained(self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.extractor = ViTFeatureExtractor.from_pretrained(self.path)

        # Move model to device
        self.model.to(self.device)

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

        # Feature extraction
        pixels = self.extractor(images=values, return_tensors="pt").pixel_values
        pixels = pixels.to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model.generate(pixels, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

        # Tokenize outputs into text results
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        captions = [caption.strip() for caption in captions]

        # Return single element if single element passed in
        return captions[0] if not isinstance(images, list) else captions

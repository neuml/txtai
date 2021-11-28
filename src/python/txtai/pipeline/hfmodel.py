"""
Hugging Face Transformers model wrapper module
"""

from ..models import Models
from .tensors import Tensors


class HFModel(Tensors):
    """
    Pipeline backed by a Hugging Face Transformers model.
    """

    def __init__(self, path=None, quantize=False, gpu=False, batch=64):
        """
        Creates a new HFModel.

        Args:
            path: optional path to model, accepts Hugging Face model hub id or local path,
                  uses default model for task if not provided
            quantize: if model should be quantized, defaults to False
            gpu: True/False if GPU should be enabled, also supports a GPU device id
            batch: batch size used to incrementally process content
        """

        # Default model path
        self.path = path

        # Quantization flag
        self.quantization = quantize

        # Get tensor device reference
        self.deviceid = Models.deviceid(gpu)
        self.device = Models.reference(self.deviceid)

        # Process batch size
        self.batchsize = batch

    def prepare(self, model):
        """
        Prepares a model for processing. Applies dynamic quantization if necessary.

        Args:
            model: input model

        Returns:
            model
        """

        if self.deviceid == -1 and self.quantization:
            model = self.quantize(model)

        return model

    def tokenize(self, tokenizer, texts):
        """
        Tokenizes text using tokenizer. This method handles overflowing tokens and automatically splits
        them into separate elements. Indices of each element is returned to allow reconstructing the
        transformed elements after running through the model.

        Args:
            tokenizer: Tokenizer
            texts: list of text

        Returns:
            (tokenization result, indices)
        """

        # Pre-process and split on newlines
        batch, positions = [], []
        for x, text in enumerate(texts):
            elements = [t + " " for t in text.split("\n") if t]
            batch.extend(elements)
            positions.extend([x] * len(elements))

        # Run tokenizer
        tokens = tokenizer(batch, padding=True)

        inputids, attention, indices = [], [], []
        for x, ids in enumerate(tokens["input_ids"]):
            if len(ids) > tokenizer.model_max_length:
                # Remove padding characters, if any
                ids = [i for i in ids if i != tokenizer.pad_token_id]

                # Split into model_max_length chunks
                for chunk in self.batch(ids, tokenizer.model_max_length - 1):
                    # Append EOS token if necessary
                    if chunk[-1] != tokenizer.eos_token_id:
                        chunk.append(tokenizer.eos_token_id)

                    # Set attention mask
                    mask = [1] * len(chunk)

                    # Append padding if necessary
                    if len(chunk) < tokenizer.model_max_length:
                        pad = tokenizer.model_max_length - len(chunk)
                        chunk.extend([tokenizer.pad_token_id] * pad)
                        mask.extend([0] * pad)

                    inputids.append(chunk)
                    attention.append(mask)
                    indices.append(positions[x])
            else:
                inputids.append(ids)
                attention.append(tokens["attention_mask"][x])
                indices.append(positions[x])

        tokens = {"input_ids": inputids, "attention_mask": attention}

        # pylint: disable=E1102
        return ({name: self.tensor(tensor).to(self.device) for name, tensor in tokens.items()}, indices)

    def batch(self, texts, size):
        """
        Splits texts into separate batch sizes specified by size.

        Args:
            texts: text elements
            size: batch size

        Returns:
            list of evenly sized batches with the last batch having the remaining elements
        """

        return [texts[x : x + size] for x in range(0, len(texts), size)]

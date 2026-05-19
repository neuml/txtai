"""
LiteRT module
"""

import os

# Conditional import
try:
    from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator
    from tokenizers import Tokenizer

    LITERT = True
except ImportError:
    LITERT = False

from ...util import Download, Library

from ..base import Vectors

# Core library imports
np = Library().numpy()


class LiteRT(Vectors):
    """
    Builds vectors using LiteRT.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is a LiteRT model.

        Args:
            path: input path

        Returns:
            True if this is a LiteRT model, False otherwise
        """

        return isinstance(path, str) and path.lower().endswith(".tflite")

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not LITERT:
            raise ImportError('LiteRT is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        # Check if this is a local path, otherwise download from the HF Hub
        model = path if os.path.exists(path) else Download()(path)

        # Also local tokenizer file
        tokenizer = os.path.dirname(path) + "/" + "tokenizer.json"
        tokenizer = tokenizer if os.path.exists(tokenizer) else Download()(tokenizer)

        # Load tokenizer and model
        tokenizer = Tokenizer.from_file(tokenizer)
        model = CompiledModel.from_file(
            model,
            HardwareAccelerator.GPU | HardwareAccelerator.NPU | HardwareAccelerator.CPU if self.config.get("gpu", True) else HardwareAccelerator.CPU,
        )

        # Calculate model batch size and max length
        buffers = model.create_input_buffers(0)
        batchsize, maxlength = buffers[0].get_tensor_details()["shape"]

        # Set maxlength on tokenizer
        tokenizer.enable_padding(length=maxlength)
        tokenizer.enable_truncation(max_length=maxlength)

        return (model, tokenizer, batchsize)

    def encode(self, data, category=None):
        # Tokenize input
        ids, masks, types = self.tokenizer(data)

        # Generate embeddings
        return self.embed(ids, masks, types)

    def tokenizer(self, data):
        """
        Tokenizes data.

        Args:
            data: batch of data

        Returns:
            tokenized data
        """

        # Unpack tokenizer
        _, tokenizer, _ = self.model

        encoding = tokenizer.encode_batch(data)
        return (
            np.array([e.ids for e in encoding], dtype=np.int32),
            np.array([e.attention_mask for e in encoding], dtype=np.int32),
            np.array([e.type_ids for e in encoding], dtype=np.int32),
        )

    def embed(self, ids, masks, types):
        """
        Generates embeddings from tokenized input.

        Args:
            ids: token ids
            masks: attention mask
            types: token type ids

        Returns:
            embeddings
        """

        # Unpack model
        model, _, batchsize = self.model

        results = []

        # Create tensor buffers
        inputs = model.create_input_buffers(0)
        outputs = model.create_output_buffers(0)

        # Create batch memory
        bids = np.zeros((batchsize, ids.shape[1]), dtype=np.int32)
        bmasks = np.zeros((batchsize, masks.shape[1]), dtype=np.int32)
        btypes = np.zeros((batchsize, types.shape[1]), dtype=np.int32)

        for start in range(0, len(ids), batchsize):
            # Batch parameters
            end = start + batchsize
            chunk = slice(start, end)
            actual = len(ids[chunk])

            # Copy batch to numpy buffer
            bids[:actual], bmasks[:actual], btypes[:actual] = ids[chunk], masks[chunk], types[chunk]

            # Copy to tensor buffer
            for x, data in enumerate([bids, bmasks, btypes]):
                inputs[x].write(data)

            # Run model
            model.run_by_index(0, inputs, outputs)

            # Get details of output buffer
            details = outputs[0].get_tensor_details()
            count = int(np.prod(details["shape"]))

            # Append result
            result = outputs[0].read(count, details["dtype"])
            results.append(result.reshape(details["shape"])[:actual])

        return np.vstack(results)

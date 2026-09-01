"""
ONNX module
"""

import os

# Conditional import
try:
    import onnxruntime as ort

    from tokenizers import Tokenizer

    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False

from ...util import Download, Library

from ..base import Vectors

# Core library imports
np = Library().numpy()


class ONNX(Vectors):
    """
    Builds vectors using ONNX Runtime.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is an ONNX model.

        Args:
            path: input path

        Returns:
            True if this is an ONNX model, False otherwise
        """

        return isinstance(path, str) and path.lower().endswith(".onnx")

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not ONNX_RUNTIME:
            raise ImportError('onnxruntime is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        # Check if this is a local path, otherwise download from the HF Hub
        model = path if os.path.exists(path) else Download()(path)

        # Create ONNX session
        session = ort.InferenceSession(model, ort.SessionOptions(), self.providers())

        # Load tokenizer. ONNX supports dynamic shapes, so pad to the longest sequence in each batch
        tokenizer = self.loadtokenizer(path)
        tokenizer.enable_padding()

        maxlength = self.config.get("maxlength")
        if maxlength:
            tokenizer.enable_truncation(max_length=maxlength)

        return (session, tokenizer)

    def loadtokenizer(self, path):
        """
        Loads the tokenizer for a model. Reads the `tokenizer` configuration option when set,
        otherwise falls back to a tokenizer.json file stored alongside the model.

        Args:
            path: model path

        Returns:
            Tokenizer
        """

        tokenizer = self.config.get("tokenizer")
        if not tokenizer:
            tokenizer = os.path.dirname(path) + "/" + "tokenizer.json"
            tokenizer = tokenizer if os.path.exists(tokenizer) else Download()(tokenizer)

        # Local tokenizer file vs a model id resolved through the HF Hub
        return Tokenizer.from_file(tokenizer) if os.path.exists(tokenizer) else Tokenizer.from_pretrained(tokenizer)

    def providers(self):
        """
        Returns a list of available and usable providers.

        Returns:
            list of available and usable providers
        """

        # Prefer the CUDA provider when it's available, it requires onnxruntime-gpu
        if self.config.get("gpu", True) and "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    def encode(self, data, category=None):
        # Unpack model
        session, tokenizer = self.model

        results = []
        for start in range(0, len(data), self.encodebatch):
            batch = data[start : start + self.encodebatch]

            inputs, masks = self.inputs(session, tokenizer.encode_batch(batch))

            # Run model and pool the outputs into a single vector per input
            results.append(self.pool(session.run(None, inputs)[0], masks))

        return np.vstack(results)

    def inputs(self, session, encoding):
        """
        Builds model inputs from a tokenized batch. Only inputs declared by the model are set,
        since not every exported model takes token_type_ids.

        Args:
            session: inference session
            encoding: tokenized batch

        Returns:
            (model inputs, attention masks)
        """

        masks = np.array([x.attention_mask for x in encoding], dtype=np.int64)
        available = {x.name for x in session.get_inputs()}

        features = {
            "input_ids": np.array([x.ids for x in encoding], dtype=np.int64),
            "attention_mask": masks,
            "token_type_ids": np.array([x.type_ids for x in encoding], dtype=np.int64),
        }

        return {name: value for name, value in features.items() if name in available}, masks

    def pool(self, outputs, masks):
        """
        Builds a single vector per input. Token-level outputs are mean pooled with the attention
        mask, outputs that are already pooled are returned as is.

        Args:
            outputs: model outputs
            masks: attention masks

        Returns:
            embeddings
        """

        # Model already returns a vector per input
        if outputs.ndim == 2:
            return outputs

        # Mean pooling, ignoring padding tokens
        masks = np.expand_dims(masks, -1).astype(outputs.dtype)
        return np.sum(outputs * masks, axis=1) / np.clip(masks.sum(axis=1), a_min=1e-9, a_max=None)

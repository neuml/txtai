"""
TextToSpeech module
"""

# Conditional import
try:
    import onnxruntime as ort

    from ttstokenizer import TTSTokenizer

    TTS = True
except ImportError:
    TTS = False

import torch
import yaml

import numpy as np

from huggingface_hub import hf_hub_download

from ..base import Pipeline


class TextToSpeech(Pipeline):
    """
    Generates speech from text.
    """

    def __init__(self, path=None, maxtokens=512):
        """
        Creates a new TextToSpeech pipeline.

        Args:
            path: optional Hugging Face model hub id
            maxtokens: maximum number of tokens model can process, defaults to 512
        """

        if not TTS:
            raise ImportError('TextToSpeech pipeline is not available - install "pipeline" extra to enable')

        # Default path
        path = path if path else "neuml/ljspeech-jets-onnx"

        # Get path to model and config
        config = hf_hub_download(path, filename="config.yaml")
        model = hf_hub_download(path, filename="model.onnx")

        # Read yaml config
        with open(config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Create tokenizer
        tokens = config.get("token", {}).get("list")
        self.tokenizer = TTSTokenizer(tokens)

        # Create ONNX Session
        self.model = ort.InferenceSession(model, ort.SessionOptions(), self.providers())

        # Max number of input tokens model can handle
        self.maxtokens = maxtokens

        # Get model input name, typically "text"
        self.input = self.model.get_inputs()[0].name

    def __call__(self, text, stream=False):
        """
        Generates speech from text. Text longer than maxtokens will be batched and returned
        as a single waveform per text input.

        This method supports files as a string or a list. If the input is a string,
        the return type is string. If text is a list, the return type is a list.

        Args:
            text: text|list
            stream: stream response if True, defaults to False

        Returns:
            list of speech as NumPy array waveforms
        """

        # Convert results to a list if necessary
        texts = [text] if isinstance(text, str) else text

        # Streaming response
        if stream:
            return self.stream(texts)

        # Transform text to speech
        outputs = [self.execute(x) for x in texts]

        # Return results
        return outputs[0] if isinstance(text, str) else outputs

    def providers(self):
        """
        Returns a list of available and usable providers.

        Returns:
            list of available and usable providers
        """

        # Create list of providers, prefer CUDA provider if available
        # CUDA provider only available if GPU is available and onnxruntime-gpu installed
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]

        # Default when CUDA provider isn't available
        return ["CPUExecutionProvider"]

    def stream(self, texts):
        """
        Iterates over texts, splits into segments and yields snippets of audio.
        This method is designed to integrate with streaming LLM generation.

        Args:
            texts: list of input texts

        Returns:
            snippets of audio as NumPy arrays
        """

        buffer = []
        for x in texts:
            buffer.append(x)

            if x == "\n" or x.strip().endswith("."):
                data, buffer = "".join(buffer), []
                yield self.execute(data)

        if buffer:
            data = "".join(buffer)
            yield self.execute(data)

    def execute(self, text):
        """
        Executes model run for an input array of tokens. This method will build batches
        of tokens when len(tokens) > maxtokens.

        Args:
            text: text to tokenize and pass to model

        Returns:
            waveform as NumPy array
        """

        # Tokenize input
        tokens = self.tokenizer(text)

        # Split into batches and process
        results = []
        for x in self.batch(tokens, self.maxtokens):
            # Run text through TTS model and save waveform
            output = self.model.run(None, {self.input: x})
            results.append(output[0])

        # Concatenate results and return
        return np.concatenate(results)

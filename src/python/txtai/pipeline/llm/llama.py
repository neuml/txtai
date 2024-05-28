"""
Llama module
"""

import os

from huggingface_hub import hf_hub_download

# Conditional import
try:
    from llama_cpp import Llama

    LLAMA_CPP = True
except ImportError:
    LLAMA_CPP = False

from .generation import Generation


class LlamaCpp(Generation):
    """
    llama.cpp generative model.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is a llama.cpp model.

        Args:
            path: input path

        Returns:
            True if this is a llama.cpp model, False otherwise
        """

        return isinstance(path, str) and path.lower().endswith(".gguf")

    def __init__(self, path, template=None, **kwargs):
        super().__init__(path, template, **kwargs)

        if not LLAMA_CPP:
            raise ImportError('llama.cpp is not available - install "pipeline" extra to enable')

        # Check if this is a local path, otherwise download from the HF Hub
        path = path if os.path.exists(path) else self.download(path)

        # Default GPU layers if not already set
        kwargs["n_gpu_layers"] = kwargs.get("n_gpu_layers", -1 if kwargs.get("gpu", os.environ.get("LLAMA_NO_METAL") != "1") else 0)

        # Create llama.cpp instance
        self.llm = Llama(path, verbose=kwargs.pop("verbose", False), **kwargs)

    def execute(self, texts, maxlength, **kwargs):
        results = []
        for text in texts:
            results.append(self.messages(text, maxlength, **kwargs) if isinstance(text, list) else self.prompt(text, maxlength, **kwargs))

        return results

    def download(self, path):
        """
        Downloads path from the Hugging Face Hub.

        Args:
            path: full model path

        Returns:
            local cached model path
        """

        # Split into parts
        parts = path.split("/")

        # Calculate repo id split
        repo = 2 if len(parts) > 2 else 1

        # Download and cache file
        return hf_hub_download(repo_id="/".join(parts[:repo]), filename="/".join(parts[repo:]))

    def messages(self, messages, maxlength, **kwargs):
        """
        Processes a list of messages.

        Args:
            messages: list of dictionaries with `role` and `content` key-values
            maxlength: maximum sequence length
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        return self.llm.create_chat_completion(messages=messages, max_tokens=maxlength, **kwargs)["choices"][0]["message"]["content"]

    def prompt(self, text, maxlength, **kwargs):
        """
        Processes a prompt.

        Args:
            prompt: prompt text
            maxlength: maximum sequence length
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        return self.llm(text, max_tokens=maxlength, **kwargs)["choices"][0]["text"]

"""
Llama module
"""

import os

from huggingface_hub import hf_hub_download

# Conditional import
try:
    import llama_cpp as llama

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

        # Create llama.cpp instance
        self.llm = self.create(path, **kwargs)

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        for text in texts:
            yield from (
                self.messages(text, maxlength, stream, stop, **kwargs)
                if isinstance(text, list)
                else self.prompt(text, maxlength, stream, stop, **kwargs)
            )

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

    def create(self, path, **kwargs):
        """
        Creates a new llama.cpp model instance.

        Args:
            path: path to model
            kwargs: additional keyword args

        Returns:
            llama.cpp instance
        """

        # Default n_ctx=0 if not already set. This sets n_ctx = n_ctx_train.
        kwargs["n_ctx"] = kwargs.get("n_ctx", 0)

        # Default GPU layers if not already set
        kwargs["n_gpu_layers"] = kwargs.get("n_gpu_layers", -1 if kwargs.get("gpu", os.environ.get("LLAMA_NO_METAL") != "1") else 0)

        # Default verbose flag
        kwargs["verbose"] = kwargs.get("verbose", False)

        # Create llama.cpp instance
        try:
            return llama.Llama(model_path=path, **kwargs)
        except ValueError as e:
            # Fallback to default n_ctx when not enough memory for n_ctx = n_ctx_train
            if not kwargs["n_ctx"]:
                kwargs.pop("n_ctx")
                return llama.Llama(model_path=path, **kwargs)

            # Raise exception if n_ctx manually specified
            raise e

    def messages(self, messages, maxlength, stream, stop, **kwargs):
        """
        Processes a list of messages.

        Args:
            messages: list of dictionaries with `role` and `content` key-values
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        # LLM call with messages
        result = self.llm.create_chat_completion(messages=messages, max_tokens=maxlength, stream=stream, stop=stop, **kwargs)

        # Stream response
        yield from self.response(result if stream else [result])

    def prompt(self, text, maxlength, stream, stop, **kwargs):
        """
        Processes a prompt.

        Args:
            prompt: prompt text
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        # LLM call with prompt
        result = self.llm(text, max_tokens=maxlength, stream=stream, stop=stop, **kwargs)

        # Stream response
        yield from self.response(result if stream else [result])

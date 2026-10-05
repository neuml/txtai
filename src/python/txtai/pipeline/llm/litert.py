"""
LiteRT module
"""

import os

# Conditional import
try:
    import litert_lm

    LITERT = True
except ImportError:
    LITERT = False

from .generation import Generation

from ...util import Download


class LiteRT(Generation):
    """
    LiteRT generative model.
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

        return isinstance(path, str) and path.lower().endswith(".litertlm")

    def __init__(self, path, template=None, **kwargs):
        super().__init__(path, template, **kwargs)

        if not LITERT:
            raise ImportError('LiteRT is not available - install "pipeline" extra to enable')

        # Set log level
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.INFO if kwargs.get("verbose") else litert_lm.LogSeverity.ERROR)

        # Check if this is a local path, otherwise download from the HF Hub
        path = path if os.path.exists(path) else Download()(path)

        # Create the engine
        self.engine = self.createengine(path, **kwargs)

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        for message in texts:
            # LiteRT only takes the last message and needs the rest as input to the conversation
            queue, message = (message[:-1], message[-1]) if len(message) > 1 else (None, message[0])

            with self.engine.create_conversation(messages=queue) as conversation:
                # LLM inference
                yield from self.response(conversation.send_message_async(message) if stream else [conversation.send_message(message)])

    def response(self, result):
        for chunk in result:
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    yield item["text"]

    def createengine(self, path, **kwargs):
        """
        Creates a new LiteRT engine.

        Args:
            path: model path
            kwargs: additional keyword args
        """

        # pylint: disable=W0702
        try:
            # GPU Backend enabled
            gpu = kwargs.get("gpu", True)

            return litert_lm.Engine(
                path,
                backend=litert_lm.Backend.GPU if gpu else litert_lm.Backend.CPU,
                enable_speculative_decoding=kwargs.get("mtp", gpu),
                max_num_tokens=kwargs.get("maxlength"),
            )

        except:
            # Fallback to CPU backend
            return litert_lm.Engine(path, backend=litert_lm.Backend.CPU)

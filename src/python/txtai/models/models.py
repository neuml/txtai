"""
Models module
"""


class Models:
    """
    Utility methods for working with machine learning models
    """

    @staticmethod
    def checklength(config, tokenizer):
        """
        Checks the length for a Hugging Face Transformers tokenizer using a Hugging Face Transformers config. Copies the
        max_position_embeddings parameter if the tokenizer has no max_length set. This helps with backwards compatibility
        with older tokenizers.

        Args:
            config: Transformers config
            tokenizer: Transformers tokenizer
        """

        # Unpack nested config, handles passing model directly
        if hasattr(config, "config"):
            config = config.config

        if hasattr(config, "max_position_embeddings") and tokenizer.model_max_length == int(1e30):
            tokenizer.model_max_length = config.max_position_embeddings

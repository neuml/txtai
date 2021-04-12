"""
Models module
"""


class Models:
    """
    Utility methods for working with machine learning models
    """

    @staticmethod
    def checklength(model, tokenizer):
        """
        Checks the length for a Hugging Face Transformers tokenizer using a Hugging Face Transformers model. Copies the
        max_position_embeddings parameter if the tokenizer has no max_length set. This helps with backwards compatibility
        with older tokenizers.

        Args:
            model: Transformers model
            tokenizer: Transformers tokenizer
        """

        if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings") and tokenizer.model_max_length == int(1e30):
            tokenizer.model_max_length = model.config.max_position_embeddings

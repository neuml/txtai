"""
Models module
"""

import torch


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

    @staticmethod
    def deviceid(gpu):
        """
        Translates a gpu flag into a device id.

        Args:
            gpu: True/False if GPU should be enabled, also supports a GPU device id

        Returns:
            device id
        """

        # Always return -1 if gpu is None or CUDA is unavailable
        if gpu is None or not torch.cuda.is_available():
            return -1

        # Default to device 0 if gpu is True and not otherwise specified
        if isinstance(gpu, bool):
            return 0 if gpu else -1

        # Return gpu as device id if gpu flag is an int
        return int(gpu)

    @staticmethod
    def device(deviceid):
        """
        Gets a tensor device.

        Args:
            deviceid: device id

        Returns:
            tensor device
        """

        # Torch device
        # pylint: disable=E1101
        return torch.device(Models.reference(deviceid))

    @staticmethod
    def reference(deviceid):
        """
        Gets a tensor device reference.

        Args:
            deviceid: device id

        Returns:
            device reference
        """

        return "cpu" if deviceid < 0 else "cuda:{}".format(deviceid)

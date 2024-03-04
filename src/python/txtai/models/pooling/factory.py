"""
Factory module
"""

import json
import os

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from .base import Pooling
from .cls import ClsPooling
from .mean import MeanPooling


class PoolingFactory:
    """
    Method to create pooling models.
    """

    @staticmethod
    def create(config):
        """
        Create a pooling model.

        Args:
            config: pooling configuration

        Returns:
            Pooling
        """

        # Unpack parameters
        path, device, tokenizer, method = config["path"], config["device"], config.get("tokenizer"), config.get("method")
        modelargs = config.get("modelargs")

        # Default pooling returns hidden state
        if isinstance(path, bytes) or (isinstance(path, str) and os.path.isfile(path)) or method == "pooling":
            return Pooling(path, device=device, tokenizer=tokenizer, modelargs=modelargs)

        # Derive pooling method if it's not specified, path is a string and path is not a local path
        if (not method or method not in ("clspooling", "meanpooling")) and (isinstance(path, str) and not os.path.exists(path)):
            method = PoolingFactory.method(path)

        # Check for cls pooling
        if method == "clspooling":
            return ClsPooling(path, device, tokenizer, modelargs=modelargs)

        # Default to mean pooling
        return MeanPooling(path, device, tokenizer, modelargs=modelargs)

    @staticmethod
    def method(path):
        """
        Determines the pooling method using the sentence transformers pooling config.

        Args:
            path: model path

        Returns:
            pooling method
        """

        # Default method
        method = "meanpooling"

        # Load 1_Pooling/config.json file and read, if available
        try:
            path = hf_hub_download(repo_id=path, filename="1_Pooling/config.json")

            with open(path, encoding="utf-8") as f:
                config = json.load(f)

                # Set to CLS pooling if it's enabled and mean pooling is disabled
                if config["pooling_mode_cls_token"] and not config["pooling_mode_mean_tokens"]:
                    method = "clspooling"

        # Ignore this error
        except EntryNotFoundError:
            pass

        return method

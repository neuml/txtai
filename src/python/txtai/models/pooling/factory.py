"""
Factory module
"""

import json
import os

from .base import Pooling
from .cls import ClsPooling
from .last import LastPooling
from .late import LatePooling
from .max import MaxPooling
from .mean import MeanPooling

from ...util import Download, DownloadError


class PoolingFactory:
    """
    Method to create pooling models.
    """

    @staticmethod
    def create(config):
        """
        Create a Pooling model.

        Args:
            config: pooling configuration

        Returns:
            Pooling
        """

        # Unpack parameters
        method, path, device, tokenizer, maxlength, loadprompts, modelargs = [
            config.get(x) for x in ["method", "path", "device", "tokenizer", "maxlength", "loadprompts", "modelargs"]
        ]

        # Derive maxlength, if applicable
        maxlength = PoolingFactory.maxlength(path) if isinstance(maxlength, bool) and maxlength else maxlength

        # Default pooling returns hidden state
        if isinstance(path, bytes) or (isinstance(path, str) and os.path.isfile(path)) or method == "pooling":
            return Pooling(path, device, tokenizer, maxlength, loadprompts, modelargs)

        # Derive pooling method if it's not specified and path is a string
        if (not method or method not in ("clspooling", "meanpooling", "maxpooling", "lastpooling", "latepooling")) and isinstance(path, str):
            method = PoolingFactory.method(path)

        # Check for cls pooling
        if method == "clspooling":
            return ClsPooling(path, device, tokenizer, maxlength, loadprompts, modelargs)

        # Check for max pooling
        if method == "maxpooling":
            return MaxPooling(path, device, tokenizer, maxlength, loadprompts, modelargs)

        # Check for last pooling
        if method == "lastpooling":
            return LastPooling(path, device, tokenizer, maxlength, loadprompts, modelargs)

        # Check for late pooling
        if method == "latepooling":
            return LatePooling(path, device, tokenizer, maxlength, loadprompts, modelargs)

        # Default to mean pooling
        return MeanPooling(path, device, tokenizer, maxlength, loadprompts, modelargs)

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

        # Load 1_Pooling/config.json file
        config = PoolingFactory.load(path, "1_Pooling/config.json")

        # Set to CLS pooling if it's enabled and mean pooling is disabled
        if config and config.get("pooling_mode_cls_token") and not config["pooling_mode_mean_tokens"]:
            method = "clspooling"

        # Set to max pooling if it's enabled and mean pooling is disabled
        if config and config.get("pooling_mode_max_tokens") and not config["pooling_mode_mean_tokens"]:
            method = "maxpooling"

        # Set to last token pooling if it's enabled and mean pooling is disabled
        if config and config.get("pooling_mode_lasttoken") and not config["pooling_mode_mean_tokens"]:
            method = "lastpooling"

        # Check for late interaction pooling
        if not config:
            # Load 1_Dense/config.json
            config = PoolingFactory.load(path, "1_Dense/config.json")
            if config:
                method = "latepooling"

            # Load config.json and check architecture
            else:
                config = PoolingFactory.load(path, "config.json")
                if config and "HF_ColBERT" in config.get("architectures", []):
                    method = "latepooling"

        return method

    @staticmethod
    def maxlength(path):
        """
        Reads the max_seq_length parameter from sentence transformers config.

        Args:
            path: model path

        Returns:
            max sequence length
        """

        # Default length is unset
        maxlength = None

        # Read max_seq_length from sentence_bert_config.json
        config = PoolingFactory.load(path, "sentence_bert_config.json")
        maxlength = config.get("max_seq_length") if config else maxlength

        return maxlength

    @staticmethod
    def load(path, name):
        """
        Loads a JSON config file from the Hugging Face Hub.

        Args:
            path: model path
            name: file to load

        Returns:
            config
        """

        # Download file and parse JSON
        config = None
        try:
            path = Download()(path, name)
            if path:
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)

        # Ignore this error - invalid repo or directory
        except DownloadError:
            pass

        return config

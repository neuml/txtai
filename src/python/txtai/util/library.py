"""
Library module
"""


# pylint: disable=C0415
class Library:
    """
    Imports core but optional dependencies with fallbacks when the library is not installed.
    """

    def arguments(self):
        """
        Imports transformers.TrainingArguments.
        """

        try:
            from transformers import TrainingArguments

        except ImportError:

            class TrainingArguments:
                """
                Stub for TrainingArguments
                """

        return TrainingArguments

    def config(self):
        """
        Imports transformers.configuration_utils.PretrainedConfig.

        Returns:
            PreTrainedConfig
        """

        try:
            from transformers.configuration_utils import PretrainedConfig

        except ImportError:

            class PretrainedConfig:
                """
                Stub for PretrainedConfig
                """

        return PretrainedConfig

    def dataset(self):
        """
        Import torch.utils.data.Dataset.

        Returns:
            Dataset
        """

        try:
            from torch.utils.data import Dataset

        except ImportError:

            class Dataset:
                """
                Stub for Dataset
                """

        return Dataset

    def hferror(self):
        """
        Import huggingface_hub.errors.HFValidationError.
        """

        try:
            from huggingface_hub.errors import HFValidationError

        except ImportError:

            class HFValidationError(Exception):
                """
                Stub for HFValidationError
                """

        return HFValidationError

    def huggingface_hub(self):
        """
        Imports huggingface_hub.

        Returns:
            torch
        """

        try:
            import huggingface_hub

        except ImportError:

            class HuggingFaceHub:
                """
                Stub for HuggingFaceHub
                """

                def __getattr__(self, name):
                    raise ImportError("Hugging Face Hub is not installed, install huggingface-hub to use this module")

            huggingface_hub = HuggingFaceHub()

        return huggingface_hub

    def model(self):
        """
        Imports transformers.modeling_utils.PreTrainedModel.

        Returns:
            PreTrainedModel
        """

        try:
            from transformers.modeling_utils import PreTrainedModel

        except ImportError:

            class PreTrainedModel:
                """
                Stub for PreTrainedModel
                """

        return PreTrainedModel

    def module(self):
        """
        Imports torch.nn.Module.

        Returns:
            Module
        """

        try:
            import torch.nn

            # pylint: disable=C0103
            Module = torch.nn.Module

        except ImportError:

            class Module:
                """
                Stub for Module
                """

        return Module

    def numpy(self):
        """
        Imports numpy.

        Returns:
            numpy
        """

        try:
            import numpy

        except ImportError:

            class NumPy:
                """
                Stub for NumPy
                """

                def __getattr__(self, name):
                    raise ImportError("NumPy is not installed, install numpy to use this module")

            numpy = NumPy()

        return numpy

    def regex(self):
        """
        Imports regex.

        Returns:
            regex
        """

        try:
            import regex

        except ImportError:

            class Regex:
                """
                Stub for Regex
                """

                def __getattr__(self, name):
                    raise ImportError("Regex is not installed, install regex to use this module")

            regex = Regex()

        return regex

    def safetensors(self):
        """
        Imports safetensors.

        Returns:
            safetensors
        """

        try:
            import safetensors

        except ImportError:

            class Safetensors:
                """
                Stub for Safetensors
                """

                def __getattr__(self, name):
                    raise ImportError("Safetensors is not installed, install safetensors to use this module")

            safetensors = Safetensors()

        return safetensors

    def torch(self):
        """
        Imports torch.

        Returns:
            torch
        """

        try:
            import torch

        except ImportError:

            class Torch:
                """
                Stub for torch
                """

                def __getattr__(self, name):
                    raise ImportError("Torch is not installed, install torch to use this module")

            torch = Torch()

        return torch

    def trainer(self):
        """
        Import transformers.Trainer
        """

        try:
            from transformers import Trainer

        except ImportError:

            class Trainer(Exception):
                """
                Stub for Trainfer
                """

        return Trainer

    def transformers(self):
        """
        Imports transformers.

        Returns:
            transformers
        """

        try:
            import transformers

        except ImportError:

            class Transformers:
                """
                Stub for transformers
                """

                def __getattr__(self, name):
                    raise ImportError("Transformers is not installed, install transformers to use this module")

            transformers = Transformers()

        return transformers

    def yaml(self):
        """
        Imports yaml.

        Returns:
            yaml
        """

        try:
            import yaml

        except ImportError:

            class YAML:
                """
                Stub for yaml
                """

                def __getattr__(self, name):
                    raise ImportError("PyYAML is not installed, install pyyaml to use this module")

            yaml = YAML()

        return yaml

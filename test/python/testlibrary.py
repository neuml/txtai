"""
Library module tests
"""

import sys
import unittest

# pylint: disable=C0415,W0611,W0621
import txtai


class TestLibrary(unittest.TestCase):
    """
    Simulates core libraries not being installed.
    """

    @classmethod
    def setUpClass(cls):
        """
        Simulates core libraries not being installed
        """

        modules = [
            "huggingface_hub",
            "huggingface_hub.errors",
            "numpy",
            "regex",
            "safetensors",
            "transformers",
            "transformers.configuration_utils",
            "transformers.modeling_utils",
            "transformers.modeling_outputs",
            "torch",
            "torch.nn",
            "torch.onnx",
            "torch.utils.data",
            "tqdm",
            "tqdm.auto",
            "yaml",
        ]

        # Get handle to all currently loaded txtai modules
        modules = modules + [key for key in sys.modules if key.startswith("txtai")]
        cls.modules = {module: None for module in modules}

        # Replace loaded modules with stubs. Save modules for later reloading
        for module in cls.modules:
            if module in sys.modules:
                cls.modules[module] = sys.modules[module]

            # Remove txtai modules. Set optional dependencies to None to prevent reloading.
            if "txtai" in module:
                if module in sys.modules:
                    del sys.modules[module]
            else:
                sys.modules[module] = None

    @classmethod
    def tearDownClass(cls):
        """
        Resets modules environment back to initial state.
        """

        # Reset replaced modules in setup
        for key, value in cls.modules.items():
            if value:
                sys.modules[key] = value
            else:
                del sys.modules[key]

    def testLibrary(self):
        """
        Test core libraries not being installed
        """

        # pylint: disable=W0106
        from txtai.util import Library

        lib = Library()

        # Test transformers stubs
        for x in [lib.arguments(), lib.config(), lib.dataset(), lib.hferror(), lib.module(), lib.model()]:
            self.assertTrue(x.__module__.endswith("library"))

        with self.assertRaises(ImportError):
            lib.huggingface_hub().hf_hub_download

        with self.assertRaises(ImportError):
            lib.numpy().dot

        with self.assertRaises(ImportError):
            lib.regex().compile

        with self.assertRaises(ImportError):
            lib.safetensors().safe_open

        with self.assertRaises(ImportError):
            lib.torch().device

        with self.assertRaises(ImportError):
            lib.tqdm().tqdm

        with self.assertRaises(ImportError):
            lib.transformers().AutoModel

        with self.assertRaises(ImportError):
            lib.yaml().safe_open

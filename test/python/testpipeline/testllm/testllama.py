"""
Llama module tests
"""

import unittest

from unittest.mock import patch

from txtai.pipeline import LLM


class TestLlama(unittest.TestCase):
    """
    llama.cpp tests.
    """

    @patch("llama_cpp.Llama")
    def testContext(self, llama):
        """
        Test n_ctx with llama.cpp
        """

        class Llama:
            """
            Mock llama.cpp instance to test invalid context
            """

            def __init__(self, **kwargs):
                if kwargs.get("n_ctx") == 0 or kwargs.get("n_ctx", 0) >= 10000:
                    raise ValueError("Failed to create context")

                # Save parameters
                self.params = kwargs

        # Mock llama.cpp instance
        llama.side_effect = Llama

        # Model to test
        path = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/tinyllama-1.1b-chat-v0.3.Q2_K.gguf"

        # Test omitting n_ctx falls back to default settings
        llm = LLM(path)
        self.assertNotIn("n_ctx", llm.generator.llm.params)

        # Test n_ctx=0 falls back to default settings
        llm = LLM(path, n_ctx=0)
        self.assertNotIn("n_ctx", llm.generator.llm.params)

        # Test n_ctx manually set
        llm = LLM(path, n_ctx=1024)
        self.assertEqual(llm.generator.llm.params["n_ctx"], 1024)

        # Mock a value for n_ctx that's too big
        with self.assertRaises(ValueError):
            llm = LLM(path, n_ctx=10000)

    def testGeneration(self):
        """
        Test generation with llama.cpp
        """

        # Test model generation with llama.cpp
        model = LLM("TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/tinyllama-1.1b-chat-v0.3.Q2_K.gguf", chat_format="chatml")

        # Test with prompt
        self.assertEqual(model("2 + 2 = ", maxlength=10, seed=0, stop=["."])[0], "4")

        # Test with list of messages
        messages = [{"role": "system", "content": "You are a helpful assistant. You answer math problems."}, {"role": "user", "content": "2+2?"}]
        self.assertIsNotNone(model(messages, maxlength=10, seed=0, stop=["."]))

        # Test default role
        self.assertIsNotNone(model("2 + 2 = ", maxlength=10, seed=0, stop=["."], defaultrole="user"))

        # Test streaming
        self.assertEqual(" ".join(x for x in model("2 + 2 = ", maxlength=10, stream=True, seed=0, stop=["."]))[0], "4")

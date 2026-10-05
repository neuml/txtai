"""
LLM module tests
"""

import unittest

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from txtai.pipeline import LLM, Generation

# pylint: disable=C0411
from utils import Utils


class TestLLM(unittest.TestCase):
    """
    LLM tests.
    """

    def testArguments(self):
        """
        Test pipeline keyword arguments
        """

        start = "Hello, how are"

        # Test that text is generated with custom parameters
        model = LLM("hf-internal-testing/tiny-random-gpt2", task="language-generation", dtype="torch.float32")
        self.assertIsNotNone(model(start))

        model = LLM("hf-internal-testing/tiny-random-gpt2", task="language-generation", dtype=torch.float32)
        self.assertIsNotNone(model(start))

    def testBatchSize(self):
        """
        Test batch size
        """

        model = LLM("sshleifer/tiny-gpt2")
        self.assertIsNotNone(model(["Hello, how are"] * 2, batch_size=2))

    def testCustom(self):
        """
        Test custom LLM framework
        """

        model = LLM("hf-internal-testing/tiny-random-gpt2", task="language-generation", method="txtai.pipeline.HFGeneration")
        self.assertIsNotNone(model("Hello, how are"))

    def testCustomInvalid(self):
        """
        Test resolving an invalid LLM framework
        """

        with self.assertRaises(ImportError):
            LLM("hf-internal-testing/tiny-random-gpt2", method="pprint.pprint")

    def testCustomNotFound(self):
        """
        Test resolving an unresolvable LLM framework
        """

        with self.assertRaises(ImportError):
            LLM("hf-internal-testing/tiny-random-gpt2", method="notfound.generation")

    def testDefaultRole(self):
        """
        Test default role
        """

        model = LLM("hf-internal-testing/tiny-random-LlamaForCausalLM")
        generator = model.generator

        # Validate that the LLM supports chat messages
        self.assertEqual(model.ischat(), True)

        messages = [
            ("Hello", list),
            ("\n<|im_start|>Hello<|im_end|>", str),
            ("<|start|>Hello<|end|>", str),
            ("<|start_of_role|>system<|end_of_role|>", str),
            ("[INST]Hello[/INST]", str),
        ]

        for message, expected in messages:
            # Test auto detection of formats
            self.assertEqual(type(generator.format([message], "auto")[0]), expected)

            # Test always setting user chat messages
            self.assertEqual(type(generator.format([message], "user")[0]), list)

            # Test always keeping as prompt text
            self.assertEqual(type(generator.format([message], "prompt")[0]), str)

    def testEmptyStream(self):
        """Return no output when the upstream stream is empty."""
        model, _ = self.streammodel([])
        self.assertEqual(list(model("question", stream=True, stripthink=True)), [])

    def testExternal(self):
        """
        Test externally loaded model
        """

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        model = LLM((model, tokenizer), template="{text}")
        start = "Hello, how are"

        # Test that text is generated
        self.assertIsNotNone(model(start))

    def testMaxLength(self):
        """
        Test max length
        """

        model = LLM("sshleifer/tiny-gpt2")
        self.assertIsInstance(model("Hello, how are", maxlength=10), str)

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        generation = Generation()
        self.assertRaises(NotImplementedError, generation.stream, None, None, None, None)

    def testPartialThinkingPrefixes(self):
        """Handle split thinking prefixes, ordinary tags, and whitespace."""
        for chunks, expected in [
            (["<", "th", "ink>", "reason", "</think>", "answer"], "answer"),
            (["<", "table>", "answer"], "<table>answer"),
            (["  ", "plain", " answer"], "plain answer"),
            (["  ", "\n"], ""),
            (["<", "t"], "<t"),
            (["<|start|>assistant<|channel|>analysis<|message|>", "reason", "<|channel|>final<|message|>answer"], "answer"),
        ]:
            with self.subTest(chunks=chunks):
                model, _ = self.streammodel(chunks)
                self.assertEqual("".join(model("question", stream=True, stripthink=True)), expected)

    def testPlainAnswerIsIncremental(self):
        """Yield ordinary text before consuming subsequent chunks."""
        model, consumed = self.streammodel(["  blue", " sky"])
        result = model("question", stream=True, stripthink=True)
        self.assertEqual(next(result), "b")
        self.assertEqual(consumed, ["  blue"])
        self.assertEqual("".join(result), "lue sky")

    def testStop(self):
        """
        Test stop strings
        """

        model = LLM("sshleifer/tiny-gpt2")
        self.assertIsNotNone(model("Hello, how are", stop=["you"]))

    def testStream(self):
        """
        Test streaming generation
        """

        model = LLM("sshleifer/tiny-gpt2")
        self.assertIsInstance(" ".join(x for x in model("Hello, how are", stream=True)), str)

    def testStripThink(self):
        """
        Test stripthink parameter
        """

        # pylint: disable=W0613
        def execute1(*args, **kwargs):
            return ["<think>test</think>you"]

        def execute2(*args, **kwargs):
            return ["<|channel|>final<|message|> you"]

        model = LLM("hf-internal-testing/tiny-random-LlamaForCausalLM")

        for method in [execute1, execute2]:
            # Override execute method
            model.generator.execute = method
            self.assertEqual(model("Hello, how are", stripthink=True), "you")
            self.assertEqual(model("Hello, how are", stripthink=False), method()[0])

    def testStripThinkStream(self):
        """
        Test stripthink parameter with streaming output
        """

        # pylint: disable=W0613
        def execute1(*args, **kwargs):
            yield from "<think>test</think>you"

        def execute2(*args, **kwargs):
            yield from "<|channel|>final<|message|>you"

        model = LLM("hf-internal-testing/tiny-random-LlamaForCausalLM")

        for method in [execute1, execute2]:
            # Override execute method
            model.generator.execute = method
            self.assertEqual("".join(model("Hello, how are", stripthink=True, stream=True)), "you")
            self.assertEqual("".join(model("Hello, how are", stripthink=False, stream=True)), "".join(list(method())))

    def testStripthinkDisabled(self):
        """Preserve chunks and consume them on demand when stripping is disabled."""
        model, consumed = self.streammodel(["<think>reason</think>", "answer"])
        result = model("question", stream=True, stripthink=False)
        self.assertEqual(next(result), "<think>reason</think>")
        self.assertEqual(consumed, ["<think>reason</think>"])
        self.assertEqual(list(result), ["answer"])

    def testVision(self):
        """
        Test vision LLM
        """

        model = LLM("neuml/tiny-random-qwen2vl")
        result = model(
            [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image", "image": Utils.PATH + "/books.jpg"}]}]
        )

        self.assertIsNotNone(result)

    def streammodel(self, chunks):
        """Create an LLM pipeline with a controlled upstream stream."""
        consumed = []

        def execute(*_args, **_kwargs):
            for chunk in chunks:
                consumed.append(chunk)
                yield chunk

        model = LLM("test", method="txtai.pipeline.Generation")
        model.generator.execute = execute
        return model, consumed

"""Test incremental output through the public LLM stream with thinking cleanup."""

import unittest

from txtai.pipeline import LLM
from txtai.pipeline.llm.generation import Generation


class StreamingGeneration(Generation):
    """Custom generation backend that records upstream chunk consumption."""

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        """Yield chunks on demand and record when each chunk is consumed."""
        for chunk in self.kwargs["chunks"]:
            self.kwargs["consumed"].append(chunk)
            yield chunk


class TestStreaming(unittest.TestCase):
    """Test public streaming output and existing thinking cleanup behavior."""

    def model(self, chunks):
        """Create a public LLM pipeline and its upstream consumption record."""
        consumed = []
        model = LLM("test", method=f"{__name__}.StreamingGeneration", chunks=chunks, consumed=consumed)
        return model, consumed

    def testPlainAnswerIsIncremental(self):
        """Yield ordinary text before consuming subsequent chunks."""
        model, consumed = self.model(["  blue", " sky"])
        result = model("question", stream=True, stripthink=True)
        self.assertEqual(next(result), "b")
        self.assertEqual(consumed, ["  blue"])
        self.assertEqual("".join(result), "lue sky")

    def testEmptyStream(self):
        """Return no output when the upstream stream is empty."""
        model, _ = self.model([])
        self.assertEqual(list(model("question", stream=True, stripthink=True)), [])

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
                model, _ = self.model(chunks)
                self.assertEqual("".join(model("question", stream=True, stripthink=True)), expected)

    def testStripthinkDisabled(self):
        """Preserve chunks and consume them on demand when stripping is disabled."""
        model, consumed = self.model(["<think>reason</think>", "answer"])
        result = model("question", stream=True, stripthink=False)
        self.assertEqual(next(result), "<think>reason</think>")
        self.assertEqual(consumed, ["<think>reason</think>"])
        self.assertEqual(list(result), ["answer"])

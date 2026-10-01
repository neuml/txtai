"""验证去除思考标签时仍按需推进公开 LLM 流。"""

import unittest

from txtai.pipeline import LLM
from txtai.pipeline.llm.generation import Generation


class StreamingGeneration(Generation):
    """记录上游消费进度的合法自定义生成后端."""

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        """按需产生片段，暴露是否过早消费后续输出."""
        for chunk in self.kwargs["chunks"]:
            self.kwargs["consumed"].append(chunk)
            yield chunk


class TestStreaming(unittest.TestCase):
    """验证公开流式入口与既有思考清理行为."""

    def model(self, chunks):
        """验证生成流的推进或清理边界."""
        consumed = []
        model = LLM("test", method=f"{__name__}.StreamingGeneration", chunks=chunks, consumed=consumed)
        return model, consumed

    def testPlainAnswerIsIncremental(self):
        """验证生成流的推进或清理边界."""
        model, consumed = self.model(["  blue", " sky"])
        result = model("question", stream=True, stripthink=True)
        self.assertEqual(next(result), "b")
        self.assertEqual(consumed, ["  blue"])
        self.assertEqual("".join(result), "lue sky")

    def testEmptyStream(self):
        """验证生成流的推进或清理边界."""
        model, _ = self.model([])
        self.assertEqual(list(model("question", stream=True, stripthink=True)), [])

    def testPartialThinkingPrefixes(self):
        """验证生成流的推进或清理边界."""
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
        """验证生成流的推进或清理边界."""
        model, consumed = self.model(["<think>reason</think>", "answer"])
        result = model("question", stream=True, stripthink=False)
        self.assertEqual(next(result), "<think>reason</think>")
        self.assertEqual(consumed, ["<think>reason</think>"])
        self.assertEqual(list(result), ["answer"])

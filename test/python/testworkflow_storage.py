"""
Storage task input handling tests.
"""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from txtai.workflow import StorageTask, Workflow


class TestStorage(unittest.TestCase):
    """
    Tests mixed storage and non-storage workflow elements.
    """

    def testNonStorageInputs(self):
        """
        Non-string data, including tagged data, passes through unchanged.
        """

        elements = [None, 0, False, {"text": "value"}, ["value"], (), ("only",), ("id", None, "tag"), ("id", 7, "tag")]
        task = StorageTask()
        with patch.object(task, "list", side_effect=AssertionError("Non-storage inputs must not be listed")):
            for element in elements:
                with self.subTest(element=element):
                    self.assertFalse(task.matches(element))
            results = task(elements)
        for source, result in zip(elements, results):
            self.assertIs(source, result)
        self.assertEqual(len(results), len(elements))

    def testMixedLocalStorage(self):
        """
        A real local listing expands normally between non-storage elements.
        """

        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory) / "notes"
            container.mkdir()
            path = container / "note.txt"
            path.write_text("example", encoding="utf-8")
            url = "local://notes"
            task = StorageTask(key=directory, secret="")
            results = list(Workflow([task], batch=2)([None, ("id", url, "tag"), 7]))
            self.assertIsNone(results[0])
            self.assertEqual(results[1], ("id", str(path), "tag"))
            self.assertEqual(results[2], 7)

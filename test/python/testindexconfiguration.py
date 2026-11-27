"""Index configuration tests."""

import os
import tempfile
import unittest

from unittest.mock import patch

from txtai.embeddings.index.configuration import Configuration


class TestConfiguration(unittest.TestCase):
    @patch.dict(os.environ, {"ALLOW_PICKLE": "True"})
    def testInterruptedSave(self):
        """An incomplete write leaves the last saved configuration readable."""

        for fmt, writer in (("json", "json.dump"), ("pickle", "txtai.serialize.pickle.Pickle.savestream")):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as path:
                saved = {"format": fmt, "version": 1}
                configuration = Configuration()
                configuration.save(saved, path)
                with patch(writer, side_effect=OSError("write interrupted")), self.assertRaises(OSError):
                    configuration.save(dict(saved, version=2), path)
                self.assertEqual(configuration.load(path), saved)

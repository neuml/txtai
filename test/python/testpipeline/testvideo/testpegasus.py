"""
Pegasus module tests
"""

import os
import unittest

from unittest.mock import MagicMock, patch

from txtai.pipeline import Pegasus


class TestPegasus(unittest.TestCase):
    """
    Pegasus video pipeline tests
    """

    @patch("txtai.pipeline.video.pegasus.TwelveLabs")
    def testAnalyze(self, client):
        """
        Test analyzing a video url
        """

        # Mock analyze response
        instance = MagicMock()
        instance.analyze.return_value.data = "A red car driving on a highway."
        client.return_value = instance

        pipeline = Pegasus(api_key="test")
        result = pipeline("https://example.com/sample.mp4", "Describe this video")

        # Single input returns a single string
        self.assertEqual(result, "A red car driving on a highway.")

        # Verify the url was wired into a video context
        kwargs = instance.analyze.call_args.kwargs
        self.assertEqual(kwargs["model_name"], "pegasus1.5")
        self.assertEqual(kwargs["video"].url, "https://example.com/sample.mp4")
        self.assertEqual(kwargs["prompt"], "Describe this video")

    @patch("txtai.pipeline.video.pegasus.TwelveLabs")
    def testAssetAndBatch(self, client):
        """
        Test analyzing an asset id and a list of videos
        """

        instance = MagicMock()
        instance.analyze.return_value.data = "summary"
        client.return_value = instance

        pipeline = Pegasus(api_key="test")
        results = pipeline([{"asset_id": "abc123"}, "https://example.com/b.mp4"], "Summarize")

        # List input returns a list
        self.assertEqual(results, ["summary", "summary"])

        # First call routed to an asset id context
        first = instance.analyze.call_args_list[0].kwargs
        self.assertEqual(first["video"].asset_id, "abc123")

    @unittest.skipIf(not os.environ.get("TWELVELABS_API_KEY"), "TWELVELABS_API_KEY not set")
    def testLive(self):
        """
        Live test - requires a TwelveLabs API key and is gated since analysis can be slow
        """

        pipeline = Pegasus()
        result = pipeline(
            "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
            "Describe what happens in this video in one sentence.",
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

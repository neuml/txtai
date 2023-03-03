"""
Cloud module tests
"""

import os
import tempfile
import time
import unittest

from unittest.mock import patch

from txtai.cloud import Cloud
from txtai.embeddings import Embeddings


class TestCloud(unittest.TestCase):
    """
    Cloud tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        # Create embeddings model, backed by sentence-transformers & transformers
        cls.embeddings = Embeddings({"format": "json", "path": "sentence-transformers/nli-mpnet-base-v2", "content": True})

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup data.
        """

        if cls.embeddings:
            cls.embeddings.close()

    def testCustom(self):
        """
        Test custom provider
        """

        # pylint: disable=E1120
        self.runHub("txtai.cloud.HuggingFaceHub")

    def testHub(self):
        """
        Test huggingface-hub integration
        """

        # pylint: disable=E1120
        self.runHub("huggingface-hub")

    def testInvalidProvider(self):
        """
        Test invalid provider identifier
        """

        # Test invalid external provider
        with self.assertRaises(ImportError):
            embeddings = Embeddings()
            embeddings.load(provider="ProviderNoExist", container="Invalid")

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        cloud = Cloud({})

        self.assertRaises(NotImplementedError, cloud.exists, None)
        self.assertRaises(NotImplementedError, cloud.load, None)
        self.assertRaises(NotImplementedError, cloud.save, None)

    def testObjectStorage(self):
        """
        Test object storage integration
        """

        # Run tests with uncompressed and compressed index
        for path in ["cloud.object", "cloud.object.tar.gz"]:
            self.runTests(path, {"provider": "local", "container": f"cloud.{time.time()}", "key": tempfile.gettempdir()})

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.get_hf_file_metadata")
    @patch("huggingface_hub.upload_file")
    @patch("huggingface_hub.create_repo")
    def runHub(self, provider, create, upload, metadata, download):
        """
        Run huggingface-hub tests. This method mocks write operations since a token won't be available.
        """

        def filemeta(url, token):
            return (url, token) if "Invalid" not in url else None

        def filedownload(**kwargs):
            if "Invalid" in kwargs["repo_id"]:
                raise FileNotFoundError

            # Return either .gitattributes file or index
            return attributes if kwargs["filename"] == ".gitattributes" else index

        # Patch write methods since token will not be available
        create.return_value = None
        upload.return_value = None
        metadata.side_effect = filemeta
        download.side_effect = filedownload

        # Create dummy index
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), f"cloud.{provider}.tar.gz")
        self.embeddings.save(index)

        # Initialize attributes file
        # pylint: disable=R1732
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
            attributes = tmp.name

        # Run tests with uncompressed and compressed index
        for path in [f"cloud.{provider}", f"cloud.{provider}.tar.gz"]:
            self.runTests(path, {"provider": provider, "container": "neuml/txtai-intro"})

    def runTests(self, path, cloud):
        """
        Runs a series of cloud sync tests.
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), path)

        # Test exists handles missing cloud storage object
        invalid = cloud.copy()
        invalid["container"] = "InvalidPathToTest"
        self.assertFalse(self.embeddings.exists(index, invalid))

        # Test exception raised when trying to load index and doesn't exist in cloud storage
        # pylint: disable=W0719
        with self.assertRaises(Exception):
            self.embeddings.load(index, invalid)

        # Save index
        self.embeddings.save(index, cloud)

        # Test object exists in cloud storage
        self.assertTrue(self.embeddings.exists(index, cloud))

        # Test object exists locally
        self.assertTrue(self.embeddings.exists(index))

        # Test index can be reloaded
        self.embeddings.load(index, cloud)

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]
        self.assertEqual(result["text"], self.data[4])

"""
Cloud module tests
"""

import os
import tempfile
import time
import unittest

from unittest.mock import Mock, patch

from huggingface_hub import hf_hub_download
from txtai.cloud import Cloud, ObjectStorage
from txtai.embeddings import Embeddings


class TestObjectStorage(unittest.TestCase):
    """
    Object storage path tests.
    """

    def testLoadPaths(self):
        """
        Test nested object paths and prefix filtering
        """

        with tempfile.TemporaryDirectory() as directory:
            cloud = ObjectStorage({"provider": "local", "container": "test", "key": directory, "prefix": "index"})
            cloud.client = Mock()
            obj = Mock()
            obj.name = "index/nested/config.json"
            container = cloud.client.get_container.return_value
            container.list_objects.return_value = [obj]

            path = os.path.join(directory, "output")
            self.assertEqual(cloud.load(path), path)
            container.list_objects.assert_called_once_with(prefix="index")
            obj.download.assert_called_once_with(os.path.realpath(os.path.join(path, obj.name)), overwrite_existing=True)
            self.assertTrue(os.path.isdir(os.path.join(path, "index", "nested")))

    def testInvalidLoadPaths(self):
        """
        Test invalid object destinations before filesystem changes
        """

        with tempfile.TemporaryDirectory() as directory:
            cloud = ObjectStorage({"provider": "local", "container": "test", "key": directory})
            cloud.client = Mock()
            obj = Mock()
            cloud.client.get_container.return_value.list_objects.return_value = [obj]
            path = os.path.join(directory, "output")
            os.makedirs(os.path.join(path, "nested"))

            for name in ["../other/file", os.path.join(directory, "other", "file"), "../output-other/file", ".", "nested"]:
                with self.subTest(name=name), patch("txtai.cloud.storage.os.makedirs") as makedirs:
                    obj.name = name
                    with self.assertRaises(IOError):
                        cloud.load(path)
                    makedirs.assert_not_called()
                    obj.download.assert_not_called()

    def testLoadLink(self):
        """
        Test object destinations through a local directory link
        """

        with tempfile.TemporaryDirectory() as directory:
            cloud = ObjectStorage({"provider": "local", "container": "test", "key": directory})
            cloud.client = Mock()
            obj = Mock()
            obj.name = "nested/config.json"
            cloud.client.get_container.return_value.list_objects.return_value = [obj]
            path = os.path.join(directory, "output")
            os.makedirs(path)
            try:
                os.symlink(directory, os.path.join(path, "nested"), target_is_directory=True)
            except OSError as e:
                if getattr(e, "winerror", None) != 1314:
                    raise
                self.skipTest("Creating symbolic links requires permission")

            with self.assertRaises(IOError):
                cloud.load(path)
            obj.download.assert_not_called()


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

        # Test invalid external providers
        with self.assertRaises(ImportError):
            embeddings = Embeddings()
            embeddings.load(provider="ProviderNoExist", container="Invalid")

        with self.assertRaises(ImportError):
            embeddings = Embeddings()
            embeddings.load(provider="pprint.pprint", container="Invalid")

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

            # Check for .gitattributes file
            if kwargs["filename"] == ".gitattributes":
                return attributes

            # Check for cloud index path
            if any(kwargs["filename"] == x for x in paths):
                return index

            # Use original method for all other requests
            return hf_hub_download(**kwargs)

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
        paths = [f"cloud.{provider}", f"cloud.{provider}.tar.gz"]
        for path in paths:
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

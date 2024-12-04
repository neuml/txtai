"""
Client module tests
"""

import os
import time
import tempfile

from txtai.embeddings import Embeddings

from .testrdbms import Common


# pylint: disable=R0904
class TestClient(Common.TestRDBMS):
    """
    Embeddings with content stored in a client RDBMS.
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

        # Content backend
        cls.backend = None

        # Create embeddings model, backed by sentence-transformers & transformers
        cls.embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup data.
        """

        if cls.embeddings:
            cls.embeddings.close()

    def setUp(self):
        """
        Set unique database path for each test.
        """

        # Generate unique database path and set on embeddings
        path = os.path.join(tempfile.gettempdir(), f"{int(time.time() * 1000)}.sqlite")
        self.backend = f"sqlite:///{path}"

        self.embeddings.config["content"] = self.backend

    def testSchema(self):
        """
        Test database creation with a specified schema
        """

        # Default sequence id
        embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=self.backend, schema="txtai")
        embeddings.index(self.data)

        result = embeddings.search("feel good story", 1)[0]
        self.assertEqual(result["text"], self.data[4])

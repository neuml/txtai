"""
Client module tests
"""

import os
import time
import tempfile
import threading

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

    def testConcurrentSearch(self):
        """
        Test that concurrent searches from multiple threads do not corrupt the shared database
        Session. Prior to the scoped_session fix, concurrent threads sharing a single Session
        would race into 'prepared'-state errors under load.
        """

        embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=self.backend)
        embeddings.index(self.data)

        # Commit the indexing transaction so all threads can see the indexed content.
        # With scoped_session each thread gets its own Session/connection, which means
        # uncommitted data from the indexing session is not visible to other threads.
        embeddings.database.connection.commit()

        errors = []
        results = []
        lock = threading.Lock()

        def search_worker():
            try:
                hits = embeddings.search("feel good story", 1)
                with lock:
                    results.append(hits[0]["text"] if hits else None)
            except Exception as ex:  # pylint: disable=broad-except
                with lock:
                    errors.append(str(ex))

        # Run 8 concurrent searches — enough to expose shared-Session corruption while
        # staying within SQLAlchemy's default QueuePool limit (pool_size=5 + max_overflow=10).
        threads = [threading.Thread(target=search_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        embeddings.close()

        # The primary assertion: no Session-state corruption errors from concurrent access.
        # ('prepared'-state / PendingRollback errors would appear here before the fix.)
        self.assertEqual(errors, [], f"Concurrent searches raised errors: {errors}")

        # All threads completed
        self.assertEqual(len(results), 8, f"Expected 8 results, got {len(results)}")

        # At least some threads returned the correct top result; a None here would indicate
        # either an empty result set (FAISS edge case) or session corruption.
        self.assertTrue(any(t == self.data[4] for t in results), f"No correct result found in {results}")

    def testSchema(self):
        """
        Test database creation with a specified schema
        """

        # Default sequence id
        embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=self.backend, schema="txtai")
        embeddings.index(self.data)

        result = embeddings.search("feel good story", 1)[0]
        self.assertEqual(result["text"], self.data[4])

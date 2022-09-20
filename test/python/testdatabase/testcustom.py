"""
Custom database tests
"""

import unittest

from txtai.database import DatabaseFactory


class TestCustom(unittest.TestCase):
    """
    Custom database backend tests.
    """

    def testCustomBackend(self):
        """
        Test resolving a custom backend
        """

        database = DatabaseFactory.create({"content": "txtai.database.SQLite"})
        self.assertIsNotNone(database)

    def testCustomBackendNotFound(self):
        """
        Test resolving an unresolvable backend
        """

        with self.assertRaises(ImportError):
            DatabaseFactory.create({"content": "notfound.database"})

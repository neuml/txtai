"""
Database tests
"""

import unittest

from txtai.database import Database


class TestDatabase(unittest.TestCase):
    """
    Base database tests.
    """

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        database = Database({})

        self.assertRaises(NotImplementedError, database.load, None)
        self.assertRaises(NotImplementedError, database.insert, None)
        self.assertRaises(NotImplementedError, database.delete, None)
        self.assertRaises(NotImplementedError, database.reindex, None)
        self.assertRaises(NotImplementedError, database.save, None)
        self.assertRaises(NotImplementedError, database.close)
        self.assertRaises(NotImplementedError, database.ids, None)
        self.assertRaises(NotImplementedError, database.count)
        self.assertRaises(NotImplementedError, database.resolve, None, None)
        self.assertRaises(NotImplementedError, database.embed, None, None)
        self.assertRaises(NotImplementedError, database.query, None, None, None, None)

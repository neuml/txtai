"""
FileDB module tests
"""

import unittest

from txtai.database import FileDB


class TestFileDB(unittest.TestCase):
    """
    Test base file database methods
    """

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        db = FileDB({})

        self.assertRaises(NotImplementedError, db.connect, None)
        self.assertRaises(NotImplementedError, db.getcursor)
        self.assertRaises(NotImplementedError, db.rows)
        self.assertRaises(NotImplementedError, db.addfunctions)
        self.assertRaises(NotImplementedError, db.copy, None)

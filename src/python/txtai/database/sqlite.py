"""
SQLite module
"""

import os
import sqlite3

from .filedb import FileDB


class SQLite(FileDB):
    """
    Database instance backed by SQLite.
    """

    def connect(self, path=""):
        return sqlite3.connect(path, check_same_thread=False)

    def getcursor(self):
        return self.connection.cursor()

    def rows(self):
        return self.cursor

    def addfunctions(self):
        if self.connection and self.functions:
            # Enable callback tracebacks to show user-defined function errors
            sqlite3.enable_callback_tracebacks(True)

            for name, argcount, fn in self.functions:
                self.connection.create_function(name, argcount, fn)

    def copy(self, path):
        # Delete existing file, if necessary
        if os.path.exists(path):
            os.remove(path)

        # Create database. Thread locking must be handled externally.
        connection = self.connect(path)

        if self.connection.in_transaction:
            # The backup call will hang if there are uncommitted changes, need to copy over
            # with iterdump (which is much slower)
            for sql in self.connection.iterdump():
                connection.execute(sql)
        else:
            # Database is up to date, can do a more efficient copy with SQLite C API
            self.connection.backup(connection)

        return connection

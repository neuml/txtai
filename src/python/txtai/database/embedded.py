"""
Embedded module
"""

from .rdbms import RDBMS


class Embedded(RDBMS):
    """
    Base class for embedded relational databases. An embedded relational database stores all content in a local file.
    """

    def __init__(self, config):
        """
        Creates a new Database.

        Args:
            config: database configuration parameters
        """

        super().__init__(config)

        # Path to database file
        self.path = None

    def load(self, path):
        # Call parent logic
        super().load(path)

        # Store path reference
        self.path = path

    def save(self, path):
        # Temporary database
        if not self.path:
            # Save temporary database
            self.connection.commit()

            # Copy data from current to new
            connection = self.copy(path)

            # Close temporary database
            self.connection.close()

            # Point connection to new connection
            self.session(connection=connection)
            self.path = path

        # Paths are equal, commit changes
        elif self.path == path:
            self.connection.commit()

        # New path is different from current path, copy data and continue using current connection
        else:
            self.copy(path).close()

    def jsonprefix(self):
        # Return json column prefix
        return "json_extract(data"

    def jsoncolumn(self, name):
        # Generate json column using json_extract function
        return f"json_extract(data, '$.{name}')"

    def copy(self, path):
        """
        Copies the current database into path.

        Args:
            path: path to write database

        Returns:
            new connection with data copied over
        """

        raise NotImplementedError

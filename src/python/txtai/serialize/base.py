"""
Serialize module
"""


class Serialize:
    """
    Base class for Serialize instances. This class serializes data to files, streams and bytes.
    """

    def load(self, path):
        """
        Loads data from path.

        Args:
            path: input path

        Returns:
            deserialized data
        """

        with open(path, "rb") as handle:
            return self.loadstream(handle)

    def save(self, data, path):
        """
        Saves data to path.

        Args:
            data: data to save
            path: output path
        """

        with open(path, "wb") as handle:
            self.savestream(data, handle)

    def loadstream(self, stream):
        """
        Loads data from stream.

        Args:
            stream: input stream

        Returns:
            deserialized data
        """

        raise NotImplementedError

    def savestream(self, data, stream):
        """
        Saves data to stream.

        Args:
            data: data to save
            stream: output stream
        """

        raise NotImplementedError

    def loadbytes(self, data):
        """
        Loads data from bytes.

        Args:
            data: input bytes

        Returns:
            deserialized data
        """

        raise NotImplementedError

    def savebytes(self, data):
        """
        Saves data as bytes.

        Args:
            data: data to save

        Returns:
            serialized data
        """

        raise NotImplementedError

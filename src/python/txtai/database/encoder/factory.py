"""
Encoder factory module
"""

from .base import Encoder


class EncoderFactory:
    """
    Encoder factory. Creates new Encoder instances.
    """

    @staticmethod
    def get(encoder):
        """
        Gets a new instance of encoder class.

        Args:
            encoder: Encoder instance class

        Returns:
            Encoder class
        """

        # Local task if no package
        if "." not in encoder:
            # Get parent package
            encoder = ".".join(__name__.split(".")[:-1]) + "." + encoder.capitalize() + "Encoder"

        parts = encoder.split(".")
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

        return m

    @staticmethod
    def create(encoder):
        """
        Creates a new Encoder instance.

        Args:
            encoder: Encoder instance class

        Returns:
            Encoder
        """

        # Return default encoder
        if encoder is True:
            return Encoder()

        # Get Encoder instance
        return EncoderFactory.get(encoder)()

"""
Encoder factory module
"""

from ...util import Resolver

from .base import Encoder
from .serialize import SerializeEncoder


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

        return Resolver()(encoder)

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

        # Supported serialization methods
        if encoder in ["messagepack", "pickle"]:
            return SerializeEncoder(encoder)

        # Get Encoder instance
        return EncoderFactory.get(encoder)()

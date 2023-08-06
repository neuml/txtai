"""
Version string
"""

import logging

# Configure logging per standard Python library recommendations
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Current version tag
__version__ = "6.0.0"

# Current pickle protocol
__pickle__ = 4

# pylint: disable=C0413
# Top-level imports, must run after variables above defined
from .app import Application
from .embeddings import Embeddings

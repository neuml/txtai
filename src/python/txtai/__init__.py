"""
Version string
"""

import logging

# Configure logging per standard Python library recommendations
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Current version tag
__version__ = "5.5.0"

# Current pickle protocol
__pickle__ = 4

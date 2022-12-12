"""
Version string
"""

import logging

# Configure logging for the txtai package
# We should use logging.basicConfig() here as it modifies the root logger.  But
# log handlers/formatters should only be tweaked in the application code.  Ref:
#   - https://stackoverflow.com/a/27017068/19357935
#   - https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Current version tag
__version__ = "5.2.0"

# Current pickle protocol
__pickle__ = 4

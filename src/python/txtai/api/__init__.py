"""
API imports
"""

# Conditional import
try:
    from .authorization import Authorization
    from .application import app, start
    from .base import API
    from .cluster import Cluster
    from .extension import Extension
    from .factory import APIFactory
    from .responses import *
    from .routers import *
    from .route import EncodingAPIRoute
except ImportError as missing:
    # pylint: disable=W0707
    raise ImportError('API is not available - install "api" extra to enable') from missing

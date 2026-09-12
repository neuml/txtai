"""
Agent imports
"""

# Conditional import
try:
    from .base import Agent
    from .factory import ProcessFactory
    from .model import PipelineModel
    from .tool import *
except ImportError as e:
    from .placeholder import Agent

    # Preserve the underlying import failure so the placeholder can surface
    # the real cause (e.g. mcpadapt/mcp incompatibility) instead of always
    # blaming a missing smolagents install. See issue #1161.
    Agent._import_error = str(e)

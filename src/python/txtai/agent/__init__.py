"""
Agent imports
"""

# Conditional import
try:
    from .base import Agent
    from .factory import ProcessFactory
    from .tool import *
except ImportError as missing:
    from .placeholder import Agent

    Agent.ERROR = missing

# smolagents is optional, it's only required by the smolagents backend
try:
    from .smolagents import PipelineModel
except ImportError:
    pass

"""
smolagents imports
"""

from .process import SmolagentsProcess

# Conditional import, these classes extend smolagents classes
try:
    from .model import PipelineModel
    from .tool import SmolagentsTool
except ImportError:
    pass

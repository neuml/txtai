"""
Base imports
"""

import logging

# Top-level imports
from .app import Application
from .embeddings import Embeddings
from .pipeline import LLM, RAG

# Configure logging per standard Python library recommendations
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

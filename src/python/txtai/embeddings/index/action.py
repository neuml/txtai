"""
Action module
"""

from enum import Enum


class Action(Enum):
    """
    Index action types
    """

    INDEX = 1
    UPSERT = 2
    REINDEX = 3

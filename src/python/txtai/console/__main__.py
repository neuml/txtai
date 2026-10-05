"""
Main module.
"""

import sys

from .base import Console


def main(path=None):
    """
    Console execution loop.

    Args:
        path: model path
    """

    Console(path).cmdloop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

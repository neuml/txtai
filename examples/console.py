"""
txtai console module.

Requires streamlit to be installed.
  pip install streamlit
"""

import sys

from cmd import Cmd

from tabulate import tabulate
from txtai.embeddings import Embeddings


class Console(Cmd):
    """
    txtai console.
    """

    def __init__(self, path):
        super().__init__()

        self.intro = "txtai console"
        self.prompt = ">>> "

        self.embeddings = None
        self.path = path

    def preloop(self):
        print(f"Loading model {self.path}")

        # Load model
        self.embeddings = Embeddings()
        self.embeddings.load(self.path)

    def default(self, line):
        # pylint: disable=W0703
        try:
            results = self.embeddings.search(line)
            print(tabulate(results, headers="keys", tablefmt="psql"))
        except Exception as ex:
            print(ex)


def main(path=None):
    """
    Console execution loop.

    Args:
        path: model path
    """

    Console(path).cmdloop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

"""
txtai console module.

Requires tabulate to be installed.
  pip install tabulate
"""

import shutil
import sys
import textwrap

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
            results = []
            maxwidth = None
            for result in self.embeddings.search(line):
                # Calculate max width using current terminal width and number of columns
                if not maxwidth:
                    maxwidth = int(shutil.get_terminal_size()[0] / len(result))

                # Wrap each value at maxwidth, if necessary
                results.append({key: self.wrap(value, maxwidth) for key, value in result.items()})

            print(tabulate(results, headers="keys", tablefmt="psql"))
        except Exception as ex:
            print(ex)

    def wrap(self, value, maxwidth):
        """
        Wraps value at maxwidth if value is a string.

        Args:
            value: input value
            maxwidth: maximum number of characters before splitting text

        Returns:
            newline wrapped text at maxwidth
        """

        return "\n".join(textwrap.wrap(value, maxwidth)) if isinstance(value, str) else value


def main(path=None):
    """
    Console execution loop.

    Args:
        path: model path
    """

    Console(path).cmdloop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

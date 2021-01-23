"""
Demo shell
"""

from cmd import Cmd

from txtai.embeddings import Embeddings


class Shell(Cmd):
    """
    Query shell.
    """

    def __init__(self):
        super().__init__()

        self.intro = "query shell"
        self.prompt = "(search) "

        self.embeddings = None
        self.data = None

    def preloop(self):
        # Create embeddings model, backed by sentence-transformers & transformers
        self.embeddings = Embeddings({"method": "transformers", "path": "sentence-transformers/bert-base-nli-mean-tokens"})

        self.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

    def default(self, line):
        # Get index of best section that best matches query
        uid = self.embeddings.similarity(line, self.data)[0][0]
        print(self.data[uid])
        print()


def main():
    """
    Shell execution loop.
    """

    Shell().cmdloop()


if __name__ == "__main__":
    main()

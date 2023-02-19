"""
Console module
"""

import os
import shlex

from cmd import Cmd

# Conditional import
try:
    from rich import box
    from rich.console import Console as RichConsole
    from rich.table import Table

    RICH = True
except ImportError:
    RICH = False

from txtai.app import Application
from txtai.embeddings import Embeddings


class Console(Cmd):
    """
    txtai console.
    """

    def __init__(self, path=None):
        """
        Creates a new command line console.

        Args:
            path: path to initial configuration, if any
        """

        super().__init__()

        if not RICH:
            raise ImportError('Console is not available - install "console" extra to enable')

        self.prompt = ">>> "

        # Rich console
        self.console = RichConsole()

        # App parameters
        self.app = None
        self.path = path

        # Parameters
        self.vhighlight = None
        self.vlimit = None

    def preloop(self):
        """
        Loads initial configuration.
        """

        self.console.print("txtai console", style="#03a9f4")

        # Load default path
        if self.path:
            self.load(self.path)

    def default(self, line):
        """
        Default event loop.

        Args:
            line: command line
        """

        # pylint: disable=W0703
        try:
            command = line.lower()
            if command.startswith(".config"):
                self.config()
            elif command.startswith(".highlight"):
                self.highlight(command)
            elif command.startswith(".limit"):
                self.limit(command)
            elif command.startswith(".load"):
                command = self.split(line)
                self.path = command[1]
                self.load(self.path)
            elif command.startswith(".workflow"):
                self.workflow(line)
            else:
                # Search is default action
                self.search(line)
        except Exception:
            self.console.print_exception()

    def config(self):
        """
        Processes .config command.
        """

        self.console.print(self.app.config)

    def highlight(self, command):
        """
        Processes .highlight command.

        Args:
            command: command line
        """

        _, action = self.split(command, "#ffff00")
        self.vhighlight = action
        self.console.print(f"Set highlight to {self.vhighlight}")

    def limit(self, command):
        """
        Processes .limit command.

        Args:
            command: command line
        """

        _, action = self.split(command, 10)
        self.vlimit = int(action)
        self.console.print(f"Set limit to {self.vlimit}")

    def load(self, path):
        """
        Processes .load command.

        Args:
            path: path to configuration
        """

        if self.isyaml(path):
            self.console.print(f"Loading application {path}")
            self.app = Application(path)
        else:
            self.console.print(f"Loading index {path}")

            # Load embeddings index
            self.app = Embeddings()
            self.app.load(path)

    def search(self, query):
        """
        Runs a search query.

        Args:
            query: query to run
        """

        if self.vhighlight:
            results = self.app.explain(query, limit=self.vlimit)
        else:
            results = self.app.search(query, limit=self.vlimit)

        columns, table = {}, Table(box=box.SQUARE, style="#03a9f4")

        # Build column list
        result = results[0]
        if isinstance(result, tuple):
            columns = dict.fromkeys(["id", "score"])
        else:
            columns = dict(result)

        # Add columns to table
        columns = list(x for x in columns if x != "tokens")
        for column in columns:
            table.add_column(column)

        # Add rows to table
        for result in results:
            if isinstance(result, tuple):
                table.add_row(*(self.render(result, None, x) for x in result))
            else:
                table.add_row(*(self.render(result, column, result.get(column)) for column in columns))

        # Print table to console
        self.console.print(table)

    def workflow(self, command):
        """
        Processes .workflow command.

        Args:
            command: command line
        """

        command = shlex.split(command)
        if isinstance(self.app, Application):
            self.console.print(list(self.app.workflow(command[1], command[2:])))

    def isyaml(self, path):
        """
        Checks if file at path is a valid YAML file.

        Args:
            path: file to check

        Returns:
            True if file is valid YAML, False otherwise
        """

        if os.path.exists(path) and os.path.isfile(path):
            try:
                return Application.read(path)
            # pylint: disable=W0702
            except:
                pass

        return False

    def split(self, command, default=None):
        """
        Splits command by whitespace.

        Args:
            command: command line
            default: default command action

        Returns:
            command action
        """

        values = command.split(" ", 1)
        return values if len(values) > 1 else (command, default)

    def render(self, result, column, value):
        """
        Renders a search result column value.

        Args:
            result: result row
            column: column name
            value: column value
        """

        if isinstance(value, float):
            return f"{value:.4f}"

        # Explain highlighting
        if column == "text" and "tokens" in result:
            spans = []
            for token, score in result["tokens"]:
                color = None
                if score >= 0.02:
                    color = f"b {self.vhighlight}"

                spans.append((token, score, color))

            if result["score"] >= 0.05 and not [color for _, _, color in spans if color]:
                mscore = max(score for _, score, _ in spans)
                spans = [(token, score, f"b {self.vhighlight}" if score == mscore else color) for token, score, color in spans]

            output = ""
            for token, _, color in spans:
                if color:
                    output += f"[{color}]{token}[/{color}] "
                else:
                    output += f"{token} "

            return output

        return str(value)

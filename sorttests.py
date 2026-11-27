"""
Sorts test methods alphabetically. This script only changes files when there is an out of 
order method.
"""

import ast
import sys
from pathlib import Path


def sorttests(path):
    """
    Sorts tests in a file.

    Args:
        path: path to file
    """

    path = Path(path)
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=str(path))

    def getstart(node):
        return min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1

    def sortclass(node):
        methods = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]

        tests = [item for item in methods if item.name.startswith("test")]

        if len(tests) < 2:
            return

        # Capture test source blocks, including decorators.
        blocks = {}
        for item in tests:
            start = getstart(item)
            end = item.end_lineno
            blocks[item] = lines[start:end]

        sortedtests = iter(sorted(tests, key=lambda item: item.name.lower()))

        # Replace only the original test positions.
        replacements = []
        for item in methods:
            if item.name.startswith("test"):
                replacements.append((getstart(item), item.end_lineno, next(sortedtests)))

        # Apply replacements from bottom to top.
        for start, end, item in reversed(replacements):
            block = blocks[item][:]

            while block and not block[-1].strip():
                block.pop()

            lines[start:end] = block

    def process(nodes):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                sortclass(node)
                process(node.body)

    process(tree.body)

    result = "".join(lines)

    if result != source:
        ast.parse(result, filename=str(path))
        path.write_text(result, encoding="utf-8")
        print(f"Sorted tests in {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sorttests.py <file1.py> [file2.py ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        try:
            sorttests(filepath)
        except (SyntaxError, ValueError, OSError) as error:
            print(f"ERROR in {filepath}: {error}")
            sys.exit(1)

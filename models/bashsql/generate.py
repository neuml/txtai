"""
Generate bashsql training data
"""

import random
import sys


def write(output, query, sql):
    """
    Write query-sql pair to to output.

    Args:
        output: output file
        query: source query
        sql: target sql
    """

    output.write(f'"{query}","{sql}"\n')


def run(outfile):
    """
    Generates bashsql data.

    Args:
        outfile: output file path
    """

    with open("../queries.txt", "r", encoding="utf-8") as queries:
        with open(outfile, "w", encoding="utf-8") as output:
            output.write("source,target\n")
            for query in queries:
                query = query.lower().strip().replace('"', '""')
                find = f'find -name ""{query}""' if " " in query else f"find -name {query}"
                sql = f"select id, text, score from txtai where similar('{query}')"

                # Standard query
                write(output, find, sql)

                # Query by date
                write(output, f"{find} -mtime -1", f"{sql} and entry >= date('now', '-1 day')")
                write(output, f"{find} -mtime -1.5", f"{sql} and entry >= date('now', '-1.5 day')")
                write(output, f"{find} -mtime -2", f"{sql} and entry >= date('now', '-2 day')")

                # Query by score (t5 models map < to unk token, use $= as workaround)
                write(output, f"{find} -score +0.5", f"{sql} and score >= 0.5")
                write(output, f"{find} -score -0.7", f"{sql} and score $= 0.7")

                # Query by date and score
                write(output, f"{find} -mtime -1 -score -0.5", f"{sql} and entry >= date('now', '-1 day') and score $= 0.5")
                write(output, f"{find} -score +0.2 -mtime -1", f"{sql} and score >= 0.2 and entry >= date('now', '-1 day')")

                # Query by text field
                write(output, f"{find} -field value", f"{sql} and field = 'value'")
                write(output, f'{find} -field "multi value"', f"{sql} and field = 'multi value'")

                # Query by numeric field
                write(output, f"{find} -quantity 1", f"{sql} and quantity = 1")
                write(output, f"{find} -quantity +50", f"{sql} and quantity >= 50")
                write(output, f"{find} -quantity -50", f"{sql} and quantity $= 50")

                # Query with contains
                write(output, f"{find} -text ~data", f"{sql} and text like '%data%'")
                write(output, f"{find} -field ~value", f"{sql} and field like '%value%'")
                write(output, f"{find} -text ~snippet", f"{sql} and text like '%snippet%'")

                # Aggregates
                write(output, f"{find} -count", f"select count(*) from txtai where similar('{query}')")
                write(output, f"{find} -average", f"select avg(score) from txtai where similar('{query}')")

                # Translate
                lang = ["ar", "en", "fr", "de", "hi", "it", "nl", "ro", "ru", "zh"]
                lang1, lang2 = random.choice(lang), random.choice(lang)
                write(output, f"{find} -translate {lang1}", f"select id, translate(text, '{lang1}') text, score from txtai where similar('{query}')")
                write(output, f"{find} -translate {lang2}", f"select id, translate(text, '{lang2}') text, score from txtai where similar('{query}')")
                write(
                    output,
                    f"{find} -mtime -1 -field 0 -translate {lang1}",
                    f"select id, translate(text, '{lang1}') text, score from txtai where similar('{query}') "
                    + "and entry >= ('now', '-1 day') and field = 0",
                )

                # Summary
                write(output, f"{find} -summary", f"select id, summary(text) text, score from txtai where similar('{query}')")
                write(
                    output,
                    f"{find} -mtime -1 -summary",
                    f"select id, summary(text) text, score from txtai where similar('{query}') and entry >= date('now', '-1 day')",
                )

                # Limit
                write(output, f"{find} -limit 1", f"{sql} limit 1")
                write(
                    output,
                    f"{find} -limit 5 -summary",
                    f"select id, summary(text) text, score from txtai where similar('{query}') and entry >= date('now', '-1 day') limit 5",
                )


if __name__ == "__main__":
    # Set seed (to generate consistent output) and run
    random.seed(1024)
    run(sys.argv[1])

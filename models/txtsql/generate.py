"""
Generate txtsql training data
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
    Generates txtsql data.

    Args:
        outfile: output file path
    """

    with open("queries.txt", "r", encoding="utf-8") as queries:
        with open(outfile, "w", encoding="utf-8") as output:
            output.write("source,target\n")
            for query in queries:
                query = query.lower().strip().replace('"', '""')
                sql = f"select id, text, score from txtai where similar('{query}')"

                # Standard query
                write(output, query, sql)

                # Query by date
                write(output, f"{query} since yesterday", f"{sql} and entry >= date('now', '-1 day')")
                write(output, f"{query} since 36 hours ago", f"{sql} and entry >= date('now', '-36 hour')")
                write(output, f"{query} over last 2 days", f"{sql} and entry >= date('now', '-2 day')")
                write(output, f"{query} since 2 days ago", f"{sql} and entry >= date('now', '-2 day')")
                write(output, f"{query} since 7 days ago", f"{sql} and entry >= date('now', '-7 day')")
                write(output, f"{query} since 2 months ago", f"{sql} and entry >= date('now', '-2 month')")

                # Query by score (t5 models map < to unk token, use $= as workaround)
                write(output, f"{query} with score greater than 0.5", f"{sql} and score >= 0.5")
                write(output, f"{query} with score less than 0.7", f"{sql} and score $= 0.7")

                # Query by date and score
                write(output, f"{query} since yesterday and score less than 0.5", f"{sql} and entry >= date('now', '-1 day') and score $= 0.5")
                write(output, f"{query} with a score greater than 0.2 since yesterday", f"{sql} and score >= 0.2 and entry >= date('now', '-1 day')")

                # Query by text field
                write(output, f"{query} with field equal to value", f"{sql} and field = 'value'")
                write(output, f"{query} with field equal to multi value", f"{sql} and field = 'multi value'")

                # Query by numeric field
                write(output, f"{query} with quantity equal to 1", f"{sql} and quantity = 1")
                write(output, f"{query} with quantity greater than 50", f"{sql} and quantity >= 50")
                write(output, f"{query} with quantity less than 50", f"{sql} and quantity $= 50")

                # Query with OR
                write(output, f"{query} having text equal data or field as snippet", f"{sql} and (text = 'data' or field = 'snippet')")
                write(output, f"{query} having text as data or field equal snippet value", f"{sql} and (text = 'data' or field = 'snippet value')")
                write(output, f"{query} with field equal snippet or text as data", f"{sql} and (field = 'snippet' or text = 'data')")

                # Query with contains
                write(output, f"{query} with data in text", f"{sql} and text like '%data%'")
                write(output, f"{query} with value in field", f"{sql} and field like '%value%'")
                write(output, f"{query} with snippet in text", f"{sql} and text like '%snippet%'")

                # Aggregates
                write(output, f"how many results are {query}", f"select count(*) from txtai where similar('{query}')")
                write(output, f"average score for {query}", f"select avg(score) from txtai where similar('{query}')")

                # Translate
                lang = ["ar", "en", "fr", "de", "hi", "it", "nl", "ro", "ru", "zh"]
                lang1, lang2 = random.choice(lang), random.choice(lang)
                write(
                    output, f"{query} translated to {lang1}", f"select id, translate(text, '{lang1}') text, score from txtai where similar('{query}')"
                )
                write(
                    output, f"{query} translated to {lang2}", f"select id, translate(text, '{lang2}') text, score from txtai where similar('{query}')"
                )

                # Summary
                write(output, f"{query} summarized", f"select id, summary(text) text, score from txtai where similar('{query}')")
                write(
                    output,
                    f"{query} since yesterday summarized",
                    f"select id, summary(text) text, score from txtai where similar('{query}') and entry >= date('now', '-1 day')",
                )


if __name__ == "__main__":
    # Set seed (to generate consistent output) and run
    random.seed(1024)
    run(sys.argv[1])

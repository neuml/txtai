"""
Search application using Open Library book data. Requires the following steps to be run:

Install Streamlit
  pip install streamlit

Download and prepare data
  mkdir openlibrary && cd openlibrary
  wget -O works.txt.gz https://openlibrary.org/data/ol_dump_works_latest.txt.gz
  gunzip works.txt.gz
  grep "\"description\":" works.txt > filtered.txt

Build index
  python books.py openlibrary

Run application
  streamlit run books.py openlibrary
"""

import json
import os
import sqlite3
import sys

import pandas as pd
import streamlit as st

from txtai.embeddings import Embeddings


class Application:
    """
    Main application.
    """

    def __init__(self, path):
        """
        Creates a new application.

        Args:
            path: root path to data
        """

        self.path = path
        self.dbpath = os.path.join(self.path, "books")

    def rows(self, index):
        """
        Iterates over dataset yielding each row.

        Args:
            index: yields rows for embeddings indexing if True, otherwise yields database rows
        """

        with open(os.path.join(self.path, "filtered.txt"), encoding="utf-8") as infile:
            for x, row in enumerate(infile):
                if x % 1000 == 0:
                    print(f"Processed {x} rows", end="\r")

                row = row.split("\t")
                uid, data = row[1], json.loads(row[4])

                description = data["description"]
                if isinstance(description, dict):
                    description = description["value"]

                if "title" in data:
                    if index:
                        yield (uid, data["title"] + ". " + description, None)
                    else:
                        cover = f"{data['covers'][0]}" if "covers" in data and data["covers"] else None
                        yield (uid, data["title"], description, cover)

    def database(self):
        """
        Builds a SQLite database.
        """

        # Database file path
        dbfile = os.path.join(self.dbpath, "books.sqlite")

        # Delete existing file
        if os.path.exists(dbfile):
            os.remove(dbfile)

        # Create output database
        db = sqlite3.connect(dbfile)

        # Create database cursor
        cur = db.cursor()

        cur.execute("CREATE TABLE books (Id TEXT PRIMARY KEY, Title TEXT, Description TEXT, Cover TEXT)")

        for uid, title, description, cover in self.rows(False):
            cur.execute("INSERT INTO books (Id, Title, Description, Cover) VALUES (?, ?, ?, ?)", (uid, title, description, cover))

        # Finish and close database
        db.commit()
        db.close()

    def build(self):
        """
        Builds an embeddings index and database.
        """

        # Build embeddings index
        embeddings = Embeddings({"path": "sentence-transformers/msmarco-distilbert-base-v4"})
        embeddings.index(self.rows(True))
        embeddings.save(self.dbpath)

        # Build SQLite DB
        self.database()

    @st.cache(allow_output_mutation=True)
    def load(self):
        """
        Loads and caches embeddings index.

        Returns:
            embeddings index
        """

        embeddings = Embeddings()
        embeddings.load(self.dbpath)

        return embeddings

    def run(self):
        """
        Runs a Streamlit application.
        """

        # Build embeddings index
        embeddings = self.load()

        db = sqlite3.connect(os.path.join(self.dbpath, "books.sqlite"))
        cur = db.cursor()

        st.title("Book search")

        st.markdown(
            "This application builds a local txtai index using book data from [openlibrary.org](https://openlibrary.org). "
            + "Links to the Open Library pages and covers are shown in the application."
        )

        query = st.text_input("Search query:")
        if query:
            ids = [uid for uid, score in embeddings.search(query, 10) if score >= 0.5]

            results = []
            for uid in ids:
                cur.execute("SELECT Title, Description, Cover FROM books WHERE Id=?", (uid,))
                result = cur.fetchone()

                if result:
                    # Build cover image
                    cover = (
                        f"<img src='http://covers.openlibrary.org/b/id/{result[2]}-M.jpg'/>"
                        if result[2]
                        else "<img src='http://openlibrary.org/images/icons/avatar_book-lg.png'/>"
                    )

                    # Append book link
                    cover = f"<a target='_blank' href='https://openlibrary.org/{uid}'>{cover}</a>"
                    title = f"<a target='_blank' href='https://openlibrary.org/{uid}'>{result[0]}</a>"

                    results.append({"Cover": cover, "Title": title, "Description": result[1]})

            st.write(pd.DataFrame(results).to_html(escape=False, index=False), unsafe_allow_html=True)

        db.close()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Application is used both to index and search
    app = Application(sys.argv[1])

    # pylint: disable=W0212
    if st._is_running_with_streamlit:
        # Run application using existing index/db
        app.run()
    else:
        # Not running through streamlit, build database/index
        app.build()

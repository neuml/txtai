"""
Application that queries Wikipedia API and summarizes the top result.

Requires streamlit to be installed.
  pip install streamlit
"""

import os
import urllib.parse

import requests
import streamlit as st

from txtai.pipeline import Summary


class Application:
    """
    Main application.
    """

    SEARCH_TEMPLATE = "https://en.wikipedia.org/w/api.php?action=opensearch&search=%s&limit=1&namespace=0&format=json"
    CONTENT_TEMPLATE = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles=%s"

    def __init__(self):
        """
        Creates a new application.
        """

        self.summary = Summary("sshleifer/distilbart-cnn-12-6")

    def run(self):
        """
        Runs a Streamlit application.
        """

        st.title("Wikipedia")
        st.markdown("This application queries the Wikipedia API and summarizes the top result.")

        query = st.text_input("Query")

        if query:
            query = urllib.parse.quote_plus(query)
            data = requests.get(Application.SEARCH_TEMPLATE % query).json()
            if data and data[1]:
                page = urllib.parse.quote_plus(data[1][0])
                content = requests.get(Application.CONTENT_TEMPLATE % page).json()
                content = list(content["query"]["pages"].values())[0]["extract"]

                st.write(self.summary(content))
                st.markdown("*Source: " + data[3][0] + "*")
            else:
                st.markdown("*No results found*")


@st.cache(allow_output_mutation=True)
def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application
    """

    return Application()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create and run application
    app = create()
    app.run()

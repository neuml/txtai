"""
Application that builds a summary of an article.

Requires streamlit to be installed.
  pip install streamlit
"""

import os

import streamlit as st

from txtai.pipeline import Summary, Textractor
from txtai.workflow import UrlTask, Task, Workflow


class Application:
    """
    Main application.
    """

    def __init__(self):
        """
        Creates a new application.
        """

        textract = Textractor(paragraphs=True, minlength=100, join=True)
        summary = Summary("sshleifer/distilbart-cnn-12-6")

        self.workflow = Workflow([UrlTask(textract), Task(summary)])

    def run(self):
        """
        Runs a Streamlit application.
        """

        st.title("Article Summary")
        st.markdown("This application builds a summary of an article.")

        url = st.text_input("URL")
        if url:
            # Run workflow and get summary
            summary = list(self.workflow([url]))[0]

            # Write results
            st.write(summary)
            st.markdown("*Source: " + url + "*")


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

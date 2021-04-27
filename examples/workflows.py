"""
Build txtai workflows.

Requires streamlit to be installed.
  pip install streamlit
"""

import os

import pandas as pd
import streamlit as st

from txtai.embeddings import Documents, Embeddings
from txtai.pipeline import Segmentation, Summary, Textractor, Transcription, Translation
from txtai.workflow import Workflow, FileTask, Task


class Application:
    """
    Streamlit application.
    """

    def __init__(self):
        """
        Creates a new Streamlit application.
        """

        # Component options
        self.components = {}

        # Defined pipelines
        self.pipelines = {}

        # Current workflow
        self.workflow = []

        # Embeddings index params
        self.embeddings = None
        self.documents = None
        self.data = None

    def number(self, label):
        """
        Extracts a number from a text input field.

        Args:
            label: label to use for text input field

        Returns:
            numeric input
        """

        value = st.sidebar.text_input(label)
        return int(value) if value else None

    def options(self, component):
        """
        Extracts component settings into a component configuration dict.

        Args:
            component: component type

        Returns:
            dict with component settings
        """

        options = {"type": component}

        st.sidebar.markdown("---")
        if component == "summary":
            st.sidebar.markdown("**Summary**  \n*Abstractive text summarization*")
            options["path"] = st.sidebar.text_input("Model", value="sshleifer/distilbart-cnn-12-6")
            options["minlength"] = self.number("Min length")
            options["maxlength"] = self.number("Max length")

        elif component in ("segment", "textract"):
            if component == "segment":
                st.sidebar.markdown("**Segment**  \n*Split text into semantic units*")
            else:
                st.sidebar.markdown("**Textractor**  \n*Extract text from documents*")

            options["sentences"] = st.sidebar.checkbox("Split sentences")
            options["lines"] = st.sidebar.checkbox("Split lines")
            options["paragraphs"] = st.sidebar.checkbox("Split paragraphs")
            options["join"] = st.sidebar.checkbox("Join tokenized")
            options["minlength"] = self.number("Min section length")

        elif component == "transcribe":
            st.sidebar.markdown("**Transcribe**  \n*Transcribe audio to text*")
            options["path"] = st.sidebar.text_input("Model", value="facebook/wav2vec2-base-960h")

        elif component == "translate":
            st.sidebar.markdown("**Translate**  \n*Machine translation*")
            options["target"] = st.sidebar.text_input("Target language code", value="en")

        elif component == "embeddings":
            st.sidebar.markdown("**Embeddings Index**  \n*Index workflow output*")
            options["path"] = st.sidebar.text_area("Embeddings model path", value="sentence-transformers/bert-base-nli-mean-tokens")

        return options

    def build(self, components):
        """
        Builds a workflow using components.

        Args:
            components: list of components to add to workflow
        """

        # Clear application
        self.__init__()

        # pylint: disable=W0108
        tasks = []
        for component in components:
            wtype = component.pop("type")
            self.components[wtype] = component

            if wtype == "summary":
                self.pipelines[wtype] = Summary(component.pop("path"))
                tasks.append(Task(lambda x: self.pipelines["summary"](x, **self.components["summary"])))

            elif wtype == "segment":
                self.pipelines[wtype] = Segmentation(**self.components["segment"])
                tasks.append(Task(self.pipelines["segment"]))

            elif wtype == "textract":
                self.pipelines[wtype] = Textractor(**self.components["textract"])
                tasks.append(FileTask(self.pipelines["textract"]))

            elif wtype == "transcribe":
                self.pipelines[wtype] = Transcription(component.pop("path"))
                tasks.append(FileTask(self.pipelines["transcribe"], r".\.wav$"))

            elif wtype == "translate":
                self.pipelines[wtype] = Translation()
                tasks.append(Task(lambda x: self.pipelines["translate"](x, **self.components["translate"])))

            elif wtype == "embeddings":
                self.embeddings = Embeddings({"method": "transformers", **component})
                self.documents = Documents()
                tasks.append(Task(self.documents.add, unpack=False))

        self.workflow = Workflow(tasks)

    def process(self, data):
        """
        Processes the current application action.

        Args:
            data: input data
        """

        if data and self.workflow:
            # Build tuples for embedding index
            if self.documents:
                data = [(x, element, None) for x, element in enumerate(data)]

            # Process workflow
            for result in self.workflow(data):
                if not self.documents:
                    st.write(result)

            # Build embeddings index
            if self.documents:
                # Cache data
                self.data = [x[1] for x in self.documents]

                with st.spinner("Building embedding index...."):
                    self.embeddings.index(self.documents)
                    self.documents.close()

                # Clear workflow
                self.documents, self.pipelines, self.workflow = None, None, None

        if self.embeddings and self.data:
            # Set query and limit
            query = st.text_input("Query")
            limit = min(5, len(self.data))

            st.markdown(
                """
            <style>
            table td:nth-child(1) {
                display: none
            }
            table th:nth-child(1) {
                display: none
            }
            table {text-align: left !important}
            </style>
            """,
                unsafe_allow_html=True,
            )

            if query:
                df = pd.DataFrame([{"content": self.data[uid], "score": score} for uid, score in self.embeddings.search(query, limit)])
                st.table(df)

    def run(self):
        """
        Runs Streamlit application.
        """

        st.sidebar.image("https://github.com/neuml/txtai/raw/master/logo.png", width=256)
        st.sidebar.markdown("# Workflow builder  \n*Build and apply workflows to data*  ")

        # Get selected components
        selected = st.sidebar.multiselect("Select components", ["embeddings", "segment", "summary", "textract", "transcribe", "translate"])

        # Get selected options
        components = [self.options(component) for component in selected]
        st.sidebar.markdown("---")

        # Build or re-build workflow when build button clicked
        build = st.sidebar.button("Build")
        if build:
            with st.spinner("Building workflow...."):
                self.build(components)

        with st.beta_expander("Data", expanded=not self.data):
            data = st.text_area("Input", height=10)

        # Parse text items
        data = [x for x in data.split("\n") if x] if "file://" in data else [data]

        # Process current action
        self.process(data)


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

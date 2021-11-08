"""
Build txtai workflows.

Requires streamlit to be installed.
  pip install streamlit
"""

import contextlib
import os
import re
import tempfile
import threading
import time

import uvicorn
import yaml

import pandas as pd
import streamlit as st

import txtai.api.application

from txtai.embeddings import Documents, Embeddings
from txtai.pipeline import Segmentation, Summary, Tabular, Textractor, Transcription, Translation
from txtai.workflow import ServiceTask, Task, UrlTask, Workflow


class Server(uvicorn.Server):
    """
    Threaded uvicorn server used to bring up an API service.
    """

    def __init__(self, application=None, host="127.0.0.1", port=8000, log_level="info"):
        """
        Initialize server configuration.
        """

        config = uvicorn.Config(application, host=host, port=port, log_level=log_level)
        super().__init__(config)

    def install_signal_handlers(self):
        """
        Signal handlers no-op.
        """

    @contextlib.contextmanager
    def service(self):
        """
        Runs threaded server service.
        """

        # pylint: disable=W0201
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield

        finally:
            self.should_exit = True
            thread.join()


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

    def split(self, text):
        """
        Splits text on commas and returns a list.

        Args:
            text: input text

        Returns:
            list
        """

        return [x.strip() for x in text.split(",")]

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

        if component == "embeddings":
            st.sidebar.markdown("**Embeddings Index**  \n*Index workflow output*")
            options["index"] = st.sidebar.text_input("Embeddings storage path")
            options["path"] = st.sidebar.text_area("Embeddings model path", value="sentence-transformers/nli-mpnet-base-v2")
            options["upsert"] = st.sidebar.checkbox("Upsert")

        elif component == "summary":
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

        elif component == "service":
            options["url"] = st.sidebar.text_input("URL")
            options["method"] = st.sidebar.selectbox("Method", ["get", "post"], index=0)
            options["params"] = st.sidebar.text_input("URL parameters")
            options["batch"] = st.sidebar.checkbox("Run as batch", value=True)
            options["extract"] = st.sidebar.text_input("Subsection(s) to extract")

            if options["params"]:
                options["params"] = {key: None for key in self.split(options["params"])}
            if options["extract"]:
                options["extract"] = self.split(options["extract"])

        elif component == "tabular":
            options["idcolumn"] = st.sidebar.text_input("Id columns")
            options["textcolumns"] = st.sidebar.text_input("Text columns")
            if options["textcolumns"]:
                options["textcolumns"] = self.split(options["textcolumns"])

        elif component == "transcribe":
            st.sidebar.markdown("**Transcribe**  \n*Transcribe audio to text*")
            options["path"] = st.sidebar.text_input("Model", value="facebook/wav2vec2-base-960h")

        elif component == "translate":
            st.sidebar.markdown("**Translate**  \n*Machine translation*")
            options["target"] = st.sidebar.text_input("Target language code", value="en")

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
            component = dict(component)
            wtype = component.pop("type")
            self.components[wtype] = component

            if wtype == "embeddings":
                self.embeddings = Embeddings({**component})
                self.documents = Documents()
                tasks.append(Task(self.documents.add, unpack=False))

            elif wtype == "segment":
                self.pipelines[wtype] = Segmentation(**self.components["segment"])
                tasks.append(Task(self.pipelines["segment"]))

            elif wtype == "service":
                tasks.append(ServiceTask(**self.components["service"]))

            elif wtype == "summary":
                self.pipelines[wtype] = Summary(component.pop("path"))
                tasks.append(Task(lambda x: self.pipelines["summary"](x, **self.components["summary"])))

            elif wtype == "tabular":
                self.pipelines[wtype] = Tabular(**self.components["tabular"])
                tasks.append(Task(self.pipelines["tabular"]))

            elif wtype == "textract":
                self.pipelines[wtype] = Textractor(**self.components["textract"])
                tasks.append(UrlTask(self.pipelines["textract"]))

            elif wtype == "transcribe":
                self.pipelines[wtype] = Transcription(component.pop("path"))
                tasks.append(UrlTask(self.pipelines["transcribe"], r".\.wav$"))

            elif wtype == "translate":
                self.pipelines[wtype] = Translation()
                tasks.append(Task(lambda x: self.pipelines["translate"](x, **self.components["translate"])))

        self.workflow = Workflow(tasks)

    def yaml(self, components):
        """
        Builds a yaml string for components.

        Args:
            components: list of components to export to YAML

        Returns:
            YAML string
        """

        # pylint: disable=W0108
        data = {}
        tasks = []
        name = None

        for component in components:
            component = dict(component)
            name = wtype = component.pop("type")

            if wtype == "summary":
                data["summary"] = {"path": component.pop("path")}
                tasks.append({"action": "summary"})

            elif wtype == "segment":
                data["segmentation"] = component
                tasks.append({"action": "segmentation"})

            elif wtype == "service":
                config = dict(**component)
                config["task"] = "service"
                tasks.append(config)

            elif wtype == "tabular":
                data["tabular"] = component
                tasks.append({"action": "tabular"})

            elif wtype == "textract":
                data["textractor"] = component
                tasks.append({"action": "textractor", "task": "url"})

            elif wtype == "transcribe":
                data["transcription"] = {"path": component.pop("path")}
                tasks.append({"action": "transcription", "task": "url"})

            elif wtype == "translate":
                data["translation"] = {}
                tasks.append({"action": "translation", "args": list(**component.values())})

            elif wtype == "embeddings":
                index = component.pop("index")
                upsert = component.pop("upsert")

                data["embeddings"] = component
                data["writable"] = True

                if index:
                    data["path"] = index

                name = "index"
                tasks.append({"action": "upsert" if upsert else "index"})

        # Add in workflow
        data["workflow"] = {name: {"tasks": tasks}}

        return (name, yaml.dump(data))

    def api(self, config):
        """
        Starts an internal uvicorn server to host an API service for the current workflow.

        Args:
            config: workflow configuration
        """

        # Generate temporary file name
        yml = os.path.join(tempfile.gettempdir(), "workflow.yml")
        with open(yml, "w") as f:
            f.write(config)

        os.environ["CONFIG"] = yml
        txtai.api.application.start()
        server = Server(txtai.api.application.app)
        with server.service():
            uid = 0
            while True:
                stop = st.empty()
                click = stop.button("stop", key=uid)
                if not click:
                    time.sleep(5)
                    uid += 1
                stop.empty()

    def find(self, key):
        """
        Lookup record from cached data by uid key.

        Args:
            key: uid to search for

        Returns:
            text for matching uid
        """

        return [text for uid, text, _ in self.data if uid == key][0]

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
                self.data = list(self.documents)

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
                df = pd.DataFrame([{"content": self.find(uid), "score": score} for uid, score in self.embeddings.search(query, limit)])
                st.table(df)

    def parse(self, data):
        """
        Parse input data, splits on new lines depending on type of tasks and format of input.

        Args:
            data: input data

        Returns:
            parsed data
        """

        if re.match(r"^(http|https|file):\/\/", data) or (self.workflow and isinstance(self.workflow.tasks[0], ServiceTask)):
            return [x for x in data.split("\n") if x]

        return [data]

    def run(self):
        """
        Runs Streamlit application.
        """

        st.sidebar.image("https://github.com/neuml/txtai/raw/master/logo.png", width=256)
        st.sidebar.markdown("# Workflow builder  \n*Build and apply workflows to data*  ")

        # Get selected components
        components = ["embeddings", "segment", "service", "summary", "tabular", "textract", "transcribe", "translate"]
        selected = st.sidebar.multiselect("Select components", components)

        # Get selected options
        components = [self.options(component) for component in selected]
        st.sidebar.markdown("---")

        with st.sidebar:
            col1, col2, col3 = st.columns(3)

            # Build or re-build workflow when build button clicked
            build = col1.button("Build", help="Build the workflow and run within this application")
            if build:
                with st.spinner("Building workflow...."):
                    self.build(components)

            # Generate API configuration
            workflow, config = self.yaml(components)

            api = col2.button("API", help="Start an API instance within this application")
            if api:
                with st.spinner("Running workflow '%s' via API service, click stop to terminate" % workflow):
                    self.api(config)

            col3.download_button("Export", config, file_name="workflow.yml", mime="text/yaml", help="Export the API workflow as YAML")

        with st.expander("Data", expanded=not self.data):
            data = st.text_area("Input", height=10)

        # Parse text items
        data = self.parse(data)

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

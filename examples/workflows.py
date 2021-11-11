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
    Main application.
    """

    def __init__(self):
        """
        Creates a new application.
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

    def load(self, components):
        """
        Load an existing workflow file.

        Args:
            components: list of components to load

        Returns:
            (names of components loaded, workflow config)
        """

        workflow = st.file_uploader("Load workflow", type=["yml"])
        if workflow:
            workflow = yaml.safe_load(workflow)

            st.markdown("---")

            # Get tasks for first workflow
            tasks = list(workflow["workflow"].values())[0]["tasks"]
            selected = []

            for task in tasks:
                name = task.get("action", task.get("task"))
                if name in components:
                    selected.append(name)
                elif name in ["index", "upsert"]:
                    selected.append("embeddings")

            return (selected, workflow)

        return (None, None)

    def state(self, key):
        """
        Lookup a session state variable.

        Args:
            key: variable key

        Returns:
            variable value
        """

        if key in st.session_state:
            return st.session_state[key]

        return None

    def appsetting(self, workflow, name):
        """
        Looks up an application configuration setting.

        Args:
            workflow: workflow configuration
            name: setting name

        Returns:
            app setting value
        """

        if workflow:
            config = workflow.get("app")
            if config:
                return config.get(name)

        return None

    def setting(self, config, name, default=None):
        """
        Looks up a component configuration setting.

        Args:
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            setting value
        """

        return config.get(name, default) if config else default

    def text(self, label, config, name, default=None):
        """
        Create a new text input field.

        Args:
            label: field label
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            text input field value
        """

        default = self.setting(config, name, default)
        if not default:
            default = ""
        elif isinstance(default, list):
            default = ",".join(default)
        elif isinstance(default, dict):
            default = ",".join(default.keys())

        return st.text_input(label, value=default)

    def number(self, label, config, name, default=None):
        """
        Creates a new numeric input field.

        Args:
            label: field label
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            numeric value
        """

        value = self.text(label, config, name, default)
        return int(value) if value else None

    def boolean(self, label, config, name, default=False):
        """
        Creates a new checkbox field.

        Args:
            label: field label
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            boolean value
        """

        default = self.setting(config, name, default)
        return st.checkbox(label, value=default)

    def select(self, label, config, name, options, default=0):
        """
        Creates a new select box field.

        Args:
            label: field label
            config: component configuration
            name: setting name
            options: list of dropdown options
            default: default setting value

        Returns:
            boolean value
        """

        index = self.setting(config, name)
        index = [x for x, option in enumerate(options) if option == default]

        # Derive default index
        default = index[0] if index else default

        return st.selectbox(label, options, index=default)

    def split(self, text):
        """
        Splits text on commas and returns a list.

        Args:
            text: input text

        Returns:
            list
        """

        return [x.strip() for x in text.split(",")]

    def options(self, component, workflow):
        """
        Extracts component settings into a component configuration dict.

        Args:
            component: component type
            workflow: existing workflow, can be None

        Returns:
            dict with component settings
        """

        # pylint: disable=R0912, R0915
        options = {"type": component}

        st.markdown("---")

        # Lookup component configuration
        #   - Runtime components have config defined within tasks
        #   - Pipeline components have config defined at workflow root
        config = None
        if workflow:
            if component in ["service", "translation"]:
                # Service config is found in tasks section
                tasks = list(workflow["workflow"].values())[0]["tasks"]
                config = [task for task in tasks if task.get("task") == component or task.get("action") == component][0]
            else:
                config = workflow.get(component)

        if component == "embeddings":
            st.markdown("**Embeddings Index**  \n*Index workflow output*")
            options["index"] = self.text("Embeddings storage path", config, "index")
            options["path"] = self.text("Embeddings model path", config, "path", "sentence-transformers/nli-mpnet-base-v2")
            options["upsert"] = self.boolean("Upsert", config, "upsert")

        elif component in ("segmentation", "textractor"):
            if component == "segmentation":
                st.markdown("**Segment**  \n*Split text into semantic units*")
            else:
                st.markdown("**Textract**  \n*Extract text from documents*")

            options["sentences"] = self.boolean("Split sentences", config, "sentences")
            options["lines"] = self.boolean("Split lines", config, "lines")
            options["paragraphs"] = self.boolean("Split paragraphs", config, "paragraphs")
            options["join"] = self.boolean("Join tokenized", config, "join")
            options["minlength"] = self.number("Min section length", config, "minlength")

        elif component == "service":
            st.markdown("**Service**  \n*Extract data from an API*")
            options["url"] = self.text("URL", config, "url")
            options["method"] = self.select("Method", config, "method", ["get", "post"], 0)
            options["params"] = self.text("URL parameters", config, "params")
            options["batch"] = self.boolean("Run as batch", config, "batch", True)
            options["extract"] = self.text("Subsection(s) to extract", config, "extract")

            if options["params"]:
                options["params"] = {key: None for key in self.split(options["params"])}
            if options["extract"]:
                options["extract"] = self.split(options["extract"])

        elif component == "summary":
            st.markdown("**Summary**  \n*Abstractive text summarization*")
            options["path"] = self.text("Model", config, "path", "sshleifer/distilbart-cnn-12-6")
            options["minlength"] = self.number("Min length", config, "minlength")
            options["maxlength"] = self.number("Max length", config, "maxlength")

        elif component == "tabular":
            st.markdown("**Tabular**  \n*Split tabular data into rows and columns*")
            options["idcolumn"] = self.text("Id columns", config, "idcolumn")
            options["textcolumns"] = self.text("Text columns", config, "textcolumns")
            if options["textcolumns"]:
                options["textcolumns"] = self.split(options["textcolumns"])

        elif component == "transcription":
            st.markdown("**Transcribe**  \n*Transcribe audio to text*")
            options["path"] = self.text("Model", config, "path", "facebook/wav2vec2-base-960h")

        elif component == "translation":
            st.markdown("**Translate**  \n*Machine translation*")
            options["target"] = self.text("Target language code", config, "args", "en")

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

            elif wtype == "segmentation":
                self.pipelines[wtype] = Segmentation(**self.components[wtype])
                tasks.append(Task(self.pipelines[wtype]))

            elif wtype == "service":
                tasks.append(ServiceTask(**self.components[wtype]))

            elif wtype == "summary":
                self.pipelines[wtype] = Summary(component.pop("path"))
                tasks.append(Task(lambda x: self.pipelines["summary"](x, **self.components["summary"])))

            elif wtype == "tabular":
                self.pipelines[wtype] = Tabular(**self.components["tabular"])
                tasks.append(Task(self.pipelines[wtype]))

            elif wtype == "textractor":
                self.pipelines[wtype] = Textractor(**self.components["textract"])
                tasks.append(UrlTask(self.pipelines[wtype]))

            elif wtype == "transcription":
                self.pipelines[wtype] = Transcription(component.pop("path"))
                tasks.append(UrlTask(self.pipelines[wtype], r".\.wav$"))

            elif wtype == "translation":
                self.pipelines[wtype] = Translation()
                tasks.append(Task(lambda x: self.pipelines["translation"](x, **self.components["translation"])))

        self.workflow = Workflow(tasks)

    def yaml(self, components):
        """
        Builds a yaml string for components.

        Args:
            components: list of components to export to YAML

        Returns:
            (workflow name, YAML string)
        """

        # pylint: disable=W0108
        data = {"app": {"data": self.state("data"), "query": self.state("query")}}
        tasks = []
        name = None

        for component in components:
            component = dict(component)
            name = wtype = component.pop("type")

            if wtype == "embeddings":
                index = component.pop("index")
                upsert = component.pop("upsert")

                data[wtype] = component
                data["writable"] = True

                if index:
                    data["path"] = index

                name = "index"
                tasks.append({"action": "upsert" if upsert else "index"})

            elif wtype == "segmentation":
                data[wtype] = component
                tasks.append({"action": wtype})

            elif wtype == "service":
                config = dict(**component)
                config["task"] = wtype
                tasks.append(config)

            elif wtype == "summary":
                data[wtype] = {"path": component.pop("path")}
                tasks.append({"action": wtype})

            elif wtype == "tabular":
                data[wtype] = component
                tasks.append({"action": wtype})

            elif wtype == "textractor":
                data[wtype] = component
                tasks.append({"action": wtype, "task": "url"})

            elif wtype == "transcription":
                data[wtype] = {"path": component.pop("path")}
                tasks.append({"action": wtype, "task": "url"})

            elif wtype == "translation":
                data[wtype] = {}
                tasks.append({"action": wtype, "args": list(component.values())})

        # Add in workflow
        data["workflow"] = {name: {"tasks": tasks}}

        return (name, yaml.dump(data))

    def api(self, config):
        """
        Starts an internal uvicorn server to host an API service for the current workflow.

        Args:
            config: workflow configuration as YAML string
        """

        # Generate workflow file
        workflow = os.path.join(tempfile.gettempdir(), "workflow.yml")
        with open(workflow, "w") as f:
            f.write(config)

        os.environ["CONFIG"] = workflow
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

    def process(self, data, workflow):
        """
        Processes the current application action.

        Args:
            data: input data
            workflow: workflow configuration
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
            default = self.appsetting(workflow, "query")
            default = default if default else ""

            # Set query and limit
            query = st.text_input("Query", value=default)
            limit = min(5, len(self.data))

            # Save query state
            st.session_state["query"] = query

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

        with st.sidebar:
            st.image("https://github.com/neuml/txtai/raw/master/logo.png", width=256)
            st.markdown("# Workflow builder  \n*Build and apply workflows to data*  ")
            st.markdown("---")

            # Component configuration
            labels = {"segmentation": "segment", "textractor": "textract", "transcription": "transcribe", "translation": "translate"}
            components = ["embeddings", "segmentation", "service", "summary", "tabular", "textractor", "transcription", "translation"]

            selected, workflow = self.load(components)
            selected = st.multiselect("Select components", components, default=selected, format_func=lambda text: labels.get(text, text))

            # Get selected options
            components = [self.options(component, workflow) for component in selected]
            st.markdown("---")

            # Export buttons
            col1, col2, col3 = st.columns(3)

            # Build or re-build workflow when build button clicked or new workflow loaded
            build = col1.button("Build", help="Build the workflow and run within this application")
            if build or (workflow and workflow != self.state("workflow")):
                with st.spinner("Building workflow...."):
                    self.build(components)

            # Generate API configuration
            name, config = self.yaml(components)

            api = col2.button("API", help="Start an API instance within this application")
            if api:
                with st.spinner("Running workflow '%s' via API service, click stop to terminate" % name):
                    self.api(config)

            col3.download_button("Export", config, file_name="workflow.yml", help="Export the API workflow as YAML")

        with st.expander("Data", expanded=not self.data):
            default = self.appsetting(workflow, "data")
            default = default if default else ""

            data = st.text_area("Input", height=10, value=default)

            # Save data and workflow state
            st.session_state["data"] = data
            st.session_state["workflow"] = workflow

        # Parse text items
        data = self.parse(data) if data else data

        # Process current action
        self.process(data, workflow)


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

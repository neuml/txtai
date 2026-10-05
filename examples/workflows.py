"""
Build txtai workflows.

Requires streamlit to be installed.
  pip install streamlit
"""

import contextlib
import copy
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
import txtai.app


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


class Process:
    """
    Container for an active Workflow process instance.
    """

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get(name, config):
        """
        Lookup or creates a new workflow process instance.

        Args:
            name: workflow name
            config: application configuration

        Returns:
            Process
        """

        process = Process()

        # Build workflow
        with st.spinner("Building workflow...."):
            process.build(name, config)

        return process

    def __init__(self):
        """
        Creates a new Process.
        """

        # Application handle
        self.application = None

        # Workflow name
        self.name = None

        # Workflow data
        self.data = None

    def build(self, name, config):
        """
        Builds an application.

        Args:
            name: workflow name
            config: application configuration
        """

        # Create application
        self.application = txtai.app.Application(config)

        # Workflow name
        self.name = name

    def run(self, data):
        """
        Runs a workflow using data as input.

        Args:
            data: input data
        """

        if data and self.application:
            # Build tuples for embedding index
            if self.application.embeddings:
                data = [(x, element, None) for x, element in enumerate(data)]

            # Process workflow
            with st.spinner("Running workflow...."):
                results = []
                for result in self.application.workflow(self.name, data):
                    # Store result
                    results.append(result)

                    # Write result if this isn't an indexing workflow
                    if not self.application.embeddings:
                        st.write(result)

                # Store workflow results
                self.data = results

    def search(self, query):
        """
        Runs a search.

        Args:
            query: input query
        """

        if self.application and query:
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

            results = []
            for result in self.application.search(query, 5):
                # Text is only present when content is stored
                if "text" not in result:
                    uid, score = result["id"], result["score"]
                    results.append({"text": self.find(uid), "score": f"{score:.2}"})
                else:
                    if "id" in result and "text" in result:
                        result["text"] = self.content(result.pop("id"), result["text"])
                    if "score" in result and result["score"]:
                        result["score"] = f'{result["score"]:.2}'

                    results.append(result)

            df = pd.DataFrame(results)
            st.write(df.to_html(escape=False), unsafe_allow_html=True)

    def find(self, key):
        """
        Lookup record from cached data by uid key.

        Args:
            key: id to search for

        Returns:
            text for matching id
        """

        # Lookup text by id
        text = [text for uid, text, _ in self.data if uid == key][0]
        return self.content(key, text)

    def content(self, uid, text):
        """
        Builds a content reference for uid and text.

        Args:
            uid: record id
            text: record text

        Returns:
            content
        """

        if uid and isinstance(uid, str) and uid.lower().startswith("http"):
            return f"<a href='{uid}' rel='noopener noreferrer' target='blank'>{text}</a>"

        return text


class Application:
    """
    Main application.
    """

    def load(self, components):
        """
        Load an existing workflow file.

        Args:
            components: list of components to load

        Returns:
            (names of components loaded, workflow config, file changed)
        """

        workflow = st.file_uploader("Load workflow", type=["yml"])
        if workflow:
            # Detect file upload change
            upload = workflow.name != self.state("path")
            st.session_state["path"] = workflow.name

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

            return (selected, workflow, upload)

        return (None, None, None)

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

    def text(self, label, component, config, name, default=None):
        """
        Create a new text input field.

        Args:
            label: field label
            component: component name
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

        return st.text_input(label, value=default, key=component + name)

    def number(self, label, component, config, name, default=None):
        """
        Creates a new numeric input field.

        Args:
            label: field label
            component: component name
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            numeric value
        """

        value = self.text(label, component, config, name, default)
        return int(value) if value else None

    def boolean(self, label, component, config, name, default=False):
        """
        Creates a new checkbox field.

        Args:
            label: field label
            component: component name
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            boolean value
        """

        default = self.setting(config, name, default)
        return st.checkbox(label, value=default, key=component + name)

    def select(self, label, component, config, name, options, default=0):
        """
        Creates a new select box field.

        Args:
            label: field label
            component: component name
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

        return st.selectbox(label, options, index=default, key=component + name)

    def split(self, text):
        """
        Splits text on commas and returns a list.

        Args:
            text: input text

        Returns:
            list
        """

        return [x.strip() for x in text.split(",")]

    def options(self, component, workflow, index):
        """
        Extracts component settings into a component configuration dict.

        Args:
            component: component type
            workflow: existing workflow, can be None
            index: task index

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
                tasks = [task for task in tasks if task.get("task") == component or task.get("action") == component]
                if tasks:
                    config = tasks[0]
            else:
                config = workflow.get(component)

        if component == "embeddings":
            st.markdown(f"**{index + 1}.) Embeddings Index**  \n*Index workflow output*")
            options["index"] = self.text("Embeddings storage path", component, config, "index")
            options["path"] = self.text("Embeddings model path", component, config, "path", "sentence-transformers/nli-mpnet-base-v2")
            options["upsert"] = self.boolean("Upsert", component, config, "upsert")
            options["content"] = self.boolean("Content", component, config, "content")

        elif component in ("segmentation", "textractor"):
            if component == "segmentation":
                st.markdown(f"**{index + 1}.) Segment**  \n*Split text into semantic units*")
            else:
                st.markdown(f"**{index + 1}.) Textract**  \n*Extract text from documents*")

            options["sentences"] = self.boolean("Split sentences", component, config, "sentences")
            options["lines"] = self.boolean("Split lines", component, config, "lines")
            options["paragraphs"] = self.boolean("Split paragraphs", component, config, "paragraphs")
            options["join"] = self.boolean("Join tokenized", component, config, "join")
            options["minlength"] = self.number("Min section length", component, config, "minlength")

        elif component == "service":
            st.markdown(f"**{index + 1}.) Service**  \n*Extract data from an API*")
            options["url"] = self.text("URL", component, config, "url")
            options["method"] = self.select("Method", component, config, "method", ["get", "post"], 0)
            options["params"] = self.text("URL parameters", component, config, "params")
            options["batch"] = self.boolean("Run as batch", component, config, "batch", True)
            options["extract"] = self.text("Subsection(s) to extract", component, config, "extract")

            if options["params"]:
                options["params"] = {key: None for key in self.split(options["params"])}
            if options["extract"]:
                options["extract"] = self.split(options["extract"])

        elif component == "summary":
            st.markdown(f"**{index + 1}.) Summary**  \n*Abstractive text summarization*")
            options["path"] = self.text("Model", component, config, "path", "sshleifer/distilbart-cnn-12-6")
            options["minlength"] = self.number("Min length", component, config, "minlength")
            options["maxlength"] = self.number("Max length", component, config, "maxlength")

        elif component == "tabular":
            st.markdown(f"**{index + 1}.) Tabular**  \n*Split tabular data into rows and columns*")
            options["idcolumn"] = self.text("Id columns", component, config, "idcolumn")
            options["textcolumns"] = self.text("Text columns", component, config, "textcolumns")
            options["content"] = self.text("Content", component, config, "content")

            if options["textcolumns"]:
                options["textcolumns"] = self.split(options["textcolumns"])

            if options["content"]:
                options["content"] = self.split(options["content"])
                if len(options["content"]) == 1 and options["content"][0] == "1":
                    options["content"] = options["content"][0]

        elif component == "transcription":
            st.markdown(f"**{index + 1}.) Transcribe**  \n*Transcribe audio to text*")
            options["path"] = self.text("Model", component, config, "path", "facebook/wav2vec2-base-960h")

        elif component == "translation":
            st.markdown(f"**{index + 1}.) Translate**  \n*Machine translation*")
            options["target"] = self.text("Target language code", component, config, "args", "en")

        return options

    def config(self, components):
        """
        Builds configuration for components

        Args:
            components: list of components to add to configuration

        Returns:
            (workflow name, configuration)
        """

        data = {}
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
                config = {**component}
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

        # Return workflow name and application configuration
        return (name, data)

    def api(self, config):
        """
        Starts an internal uvicorn server to host an API service for the current workflow.

        Args:
            config: workflow configuration as YAML string
        """

        # Generate workflow file
        workflow = os.path.join(tempfile.gettempdir(), "workflow.yml")
        with open(workflow, "w", encoding="utf-8") as f:
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

    def inputs(self, selected, workflow):
        """
        Generate process input fields.

        Args:
            selected: list of selected components
            workflow: workflow configuration

        Returns:
            True if inputs changed, False otherwise
        """

        change, query = False, None
        with st.expander("Data", expanded="embeddings" not in selected):
            default = self.appsetting(workflow, "data")
            default = default if default else ""

            data = st.text_area("Input", height=10, value=default)

            if selected and data and data != self.state("data"):
                change = True

            # Save data and workflow state
            st.session_state["data"] = data

        if "embeddings" in selected:
            default = self.appsetting(workflow, "query")
            default = default if default else ""

            # Set query and limit
            query = st.text_input("Query", value=default)

            if selected and query and query != self.state("query"):
                change = True

        # Save query state
        st.session_state["query"] = query

        return change or self.state("api") or self.state("download")

    def data(self):
        """
        Gets input data.

        Returns:
            input data
        """

        data = self.state("data")

        # Split on newlines if urls detected, allows a list of urls to be processed
        if re.match(r"^(http|https|file):\/\/", data):
            return [x for x in data.split("\n") if x]

        return [data]

    def process(self, components, index):
        """
        Processes the current application action.

        Args:
            components: workflow components
            index: True if this is an indexing workflow
        """

        # Generate application configuration
        name, config = self.config(components)

        # Get workflow process
        process = Process.get(name, copy.deepcopy(config))

        # Run workflow process
        process.run(self.data())

        # Run search
        if index:
            process.search(self.state("query"))

        return name, config

    def run(self):
        """
        Runs Streamlit application.
        """

        build = False
        with st.sidebar:
            st.image("https://github.com/neuml/txtai/raw/master/logo.png", width=256)
            st.markdown("# Workflow builder  \n*Build and apply workflows to data*  ")
            st.markdown("---")

            # Component configuration
            labels = {"segmentation": "segment", "textractor": "textract", "transcription": "transcribe", "translation": "translate"}
            components = ["embeddings", "segmentation", "service", "summary", "tabular", "textractor", "transcription", "translation"]

            selected, workflow, upload = self.load(components)
            selected = st.multiselect("Select components", components, default=selected, format_func=lambda text: labels.get(text, text))

            if selected:
                st.markdown(
                    """
                <style>
                [data-testid="stForm"] {
                    border: 0;
                    padding: 0;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )

                with st.form("workflow"):
                    # Get selected options
                    components = [self.options(component, workflow, x) for x, component in enumerate(selected)]
                    st.markdown("---")

                    # Build or re-build workflow when build button clicked or new workflow loaded
                    build = st.form_submit_button("Build", help="Build the workflow and run within this application")

        # Generate input fields
        inputs = self.inputs(selected, workflow)

        # Only execute if build button clicked, new workflow uploaded or inputs changed
        if build or upload or inputs:
            # Process current action
            name, config = self.process(components, "embeddings" in selected)

            with st.sidebar:
                with st.expander("Other Actions", expanded=True):
                    col1, col2 = st.columns(2)

                    # Add state information to configuration and export to YAML string
                    config = config.copy()
                    config.update({"app": {"data": self.state("data"), "query": self.state("query")}})
                    config = yaml.dump(config)

                    api = col1.button("API", key="api", help="Start an API instance within this application")
                    if api:
                        with st.spinner(f"Running workflow '{name}' via API service, click stop to terminate"):
                            self.api(config)

                    col2.download_button("Export", config, file_name="workflow.yml", key="download", help="Export the API workflow as YAML")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create and run application
    app = Application()
    app.run()

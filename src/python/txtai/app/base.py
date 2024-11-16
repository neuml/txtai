"""
Application module
"""

import os

from multiprocessing.pool import ThreadPool
from threading import RLock

import yaml

from ..agent import Agent
from ..embeddings import Documents, Embeddings
from ..pipeline import PipelineFactory
from ..workflow import WorkflowFactory


# pylint: disable=R0904
class Application:
    """
    Builds YAML-configured txtai applications.
    """

    @staticmethod
    def read(data):
        """
        Reads a YAML configuration file.

        Args:
            data: input data

        Returns:
            yaml
        """

        if isinstance(data, str):
            if os.path.exists(data):
                # Read yaml from file
                with open(data, "r", encoding="utf-8") as f:
                    # Read configuration
                    return yaml.safe_load(f)

            # Attempt to read yaml from input
            data = yaml.safe_load(data)
            if not isinstance(data, str):
                return data

            # File not found and input is not yaml, raise error
            raise FileNotFoundError(f"Unable to load file '{data}'")

        # Return unmodified
        return data

    def __init__(self, config, loaddata=True):
        """
        Creates an Application instance, which encapsulates embeddings, pipelines and workflows.

        Args:
            config: index configuration
            loaddata: If True (default), load existing index data, if available. Otherwise, only load models.
        """

        # Initialize member variables
        self.config, self.documents, self.embeddings = Application.read(config), None, None

        # Write lock - allows only a single thread to update embeddings
        self.lock = RLock()

        # ThreadPool - runs scheduled workflows
        self.pool = None

        # Create pipelines
        self.createpipelines()

        # Create workflows
        self.createworkflows()

        # Create agents
        self.createagents()

        # Create embeddings index
        self.indexes(loaddata)

    def __del__(self):
        """
        Close threadpool when this object is garbage collected.
        """

        if hasattr(self, "pool") and self.pool:
            self.pool.close()
            self.pool = None

    def createpipelines(self):
        """
        Create pipelines.
        """

        # Pipeline definitions
        self.pipelines = {}

        # Default pipelines
        pipelines = list(PipelineFactory.list().keys())

        # Add custom pipelines
        for key in self.config:
            if "." in key:
                pipelines.append(key)

        # Move dependent pipelines to end of list
        dependent = ["similarity", "extractor", "rag"]
        pipelines = sorted(pipelines, key=lambda x: dependent.index(x) + 1 if x in dependent else 0)

        # Create pipelines
        for pipeline in pipelines:
            if pipeline in self.config:
                config = self.config[pipeline] if self.config[pipeline] else {}

                # Add application reference, if requested
                if "application" in config:
                    config["application"] = self

                # Custom pipeline parameters
                if pipeline in ["extractor", "rag"]:
                    if "similarity" not in config:
                        # Add placeholder, will be set to embeddings index once initialized
                        config["similarity"] = None

                    # Resolve reference pipelines
                    if config.get("similarity") in self.pipelines:
                        config["similarity"] = self.pipelines[config["similarity"]]

                    if config.get("path") in self.pipelines:
                        config["path"] = self.pipelines[config["path"]]

                elif pipeline == "similarity" and "path" not in config and "labels" in self.pipelines:
                    config["model"] = self.pipelines["labels"]

                self.pipelines[pipeline] = PipelineFactory.create(config, pipeline)

    def createworkflows(self):
        """
        Create workflows.
        """

        # Workflow definitions
        self.workflows = {}

        # Create workflows
        if "workflow" in self.config:
            for workflow, config in self.config["workflow"].items():
                # Create copy of config
                config = config.copy()

                # Resolve callable functions
                config["tasks"] = [self.resolvetask(task) for task in config["tasks"]]

                # Resolve stream functions
                if "stream" in config:
                    config["stream"] = self.resolvetask(config["stream"])

                # Get scheduler config
                schedule = config.pop("schedule", None)

                # Create workflow
                self.workflows[workflow] = WorkflowFactory.create(config, workflow)

                # Schedule job if necessary
                if schedule:
                    # Create pool if necessary
                    if not self.pool:
                        self.pool = ThreadPool()

                    self.pool.apply_async(self.workflows[workflow].schedule, kwds=schedule)

    def createagents(self):
        """
        Create agents.
        """

        # Agent definitions
        self.agents = {}

        # Create agents
        if "agent" in self.config:
            for agent, config in self.config["agent"].items():
                # Create copy of config
                config = config.copy()

                # Resolve LLM
                config["llm"] = self.function("llm")

                # Resolve tools
                for tool in config.get("tools", []):
                    if isinstance(tool, dict) and "target" in tool:
                        tool["target"] = self.function(tool["target"])

                # Create agent
                self.agents[agent] = Agent(**config)

    def indexes(self, loaddata):
        """
        Initialize an embeddings index.

        Args:
            loaddata: If True (default), load existing index data, if available. Otherwise, only load models.
        """

        # Get embeddings configuration
        config = self.config.get("embeddings")
        if config:
            # Resolve application functions in embeddings config
            config = self.resolveconfig(config.copy())

        # Load embeddings index if loaddata and index exists
        if loaddata and Embeddings().exists(self.config.get("path"), self.config.get("cloud")):
            # Initialize empty embeddings
            self.embeddings = Embeddings()

            # Pass path and cloud settings. Set application functions as config overrides.
            self.embeddings.load(
                self.config.get("path"),
                self.config.get("cloud"),
                {key: config[key] for key in ["functions", "transform"] if key in config} if config else None,
            )

        elif "embeddings" in self.config:
            # Create new embeddings with config
            self.embeddings = Embeddings(config)

        # If an extractor pipeline is defined and the similarity attribute is None, set to embeddings index
        for key in ["extractor", "rag"]:
            pipeline = self.pipelines.get(key)
            config = self.config.get(key)

            if pipeline and config is not None and config["similarity"] is None:
                pipeline.similarity = self.embeddings

    def resolvetask(self, task):
        """
        Resolves callable functions for a task.

        Args:
            task: input task config
        """

        # Check for task shorthand syntax
        task = {"action": task} if isinstance(task, (str, list)) else task

        if "action" in task:
            action = task["action"]
            values = [action] if not isinstance(action, list) else action

            actions = []
            for a in values:
                if a in ["index", "upsert"]:
                    # Add queue action to buffer documents to index
                    actions.append(self.add)

                    # Override and disable unpacking for indexing actions
                    task["unpack"] = False

                    # Add finalize to trigger indexing
                    task["finalize"] = self.upsert if a == "upsert" else self.index
                elif a == "search":
                    actions.append(self.batchsearch)
                elif a == "transform":
                    # Transform vectors
                    actions.append(self.batchtransform)

                    # Override and disable one-to-many transformations
                    task["onetomany"] = False
                else:
                    # Resolve action to callable function
                    actions.append(self.function(a))

            # Save resolved action(s)
            task["action"] = actions[0] if not isinstance(action, list) else actions

        # Resolve initializer
        if "initialize" in task and isinstance(task["initialize"], str):
            task["initialize"] = self.function(task["initialize"])

        # Resolve finalizer
        if "finalize" in task and isinstance(task["finalize"], str):
            task["finalize"] = self.function(task["finalize"])

        return task

    def resolveconfig(self, config):
        """
        Resolves callable functions stored in embeddings configuration.

        Args:
            config: embeddings config

        Returns:
            resolved config
        """

        if "functions" in config:
            # Resolve callable functions
            functions = []
            for fn in config["functions"]:
                original = fn
                try:
                    if isinstance(fn, dict):
                        fn = fn.copy()
                        fn["function"] = self.function(fn["function"])
                    else:
                        fn = self.function(fn)

                # pylint: disable=W0703
                except Exception:
                    # Not a resolvable function, pipeline or workflow - further resolution will happen in embeddings
                    fn = original

                functions.append(fn)

            config["functions"] = functions

        if "transform" in config:
            # Resolve transform function
            config["transform"] = self.function(config["transform"])

        return config

    def function(self, function):
        """
        Get a handle to a callable function.

        Args:
            function: function name

        Returns:
            resolved function
        """

        # Check if function is a pipeline
        if function in self.pipelines:
            return self.pipelines[function]

        # Check if function is a workflow
        if function in self.workflows:
            return self.workflows[function]

        # Attempt to resolve action as a callable function
        return PipelineFactory.create({}, function)

    def search(self, query, limit=10, weights=None, index=None, parameters=None, graph=False):
        """
        Finds documents most similar to the input query. This method will run either an index search
        or an index + database search depending on if a database is available.

        Args:
            query: input query
            limit: maximum results
            weights: hybrid score weights, if applicable
            index: index name, if applicable
            parameters: dict of named parameters to bind to placeholders
            graph: return graph results if True

        Returns:
            list of {id: value, score: value} for index search, list of dict for an index + database search
        """

        if self.embeddings:
            with self.lock:
                results = self.embeddings.search(query, limit, weights, index, parameters, graph)

            # Unpack (id, score) tuple, if necessary. Otherwise, results are dictionaries.
            return results if graph else [{"id": r[0], "score": float(r[1])} if isinstance(r, tuple) else r for r in results]

        return None

    def batchsearch(self, queries, limit=10, weights=None, index=None, parameters=None, graph=False):
        """
        Finds documents most similar to the input queries. This method will run either an index search
        or an index + database search depending on if a database is available.

        Args:
            queries: input queries
            limit: maximum results
            weights: hybrid score weights, if applicable
            index: index name, if applicable
            parameters: list of dicts of named parameters to bind to placeholders
            graph: return graph results if True

        Returns:
            list of {id: value, score: value} per query for index search, list of dict per query for an index + database search
        """

        if self.embeddings:
            with self.lock:
                search = self.embeddings.batchsearch(queries, limit, weights, index, parameters, graph)

            results = []
            for result in search:
                # Unpack (id, score) tuple, if necessary. Otherwise, results are dictionaries.
                results.append(result if graph else [{"id": r[0], "score": float(r[1])} if isinstance(r, tuple) else r for r in result])
            return results

        return None

    def add(self, documents):
        """
        Adds a batch of documents for indexing.

        Args:
            documents: list of {id: value, data: value, tags: value}

        Returns:
            unmodified input documents
        """

        # Raise error if index is not writable
        if not self.config.get("writable"):
            raise ReadOnlyError("Attempting to add documents to a read-only index (writable != True)")

        if self.embeddings:
            with self.lock:
                # Create documents file if not already open
                if not self.documents:
                    self.documents = Documents()

                # Add documents
                self.documents.add(list(documents))

        # Return unmodified input documents
        return documents

    def addobject(self, data, uid, field):
        """
        Helper method that builds a batch of object documents.

        Args:
            data: object content
            uid: optional list of corresponding uids
            field: optional field to set

        Returns:
            documents
        """

        # Raise error if index is not writable
        if not self.config.get("writable"):
            raise ReadOnlyError("Attempting to add documents to a read-only index (writable != True)")

        documents = []
        for x, content in enumerate(data):
            if field:
                row = {"id": uid[x], field: content} if uid else {field: content}
            elif uid:
                row = (uid[x], content)
            else:
                row = content

            documents.append(row)

        return self.add(documents)

    def index(self):
        """
        Builds an embeddings index for previously batched documents.
        """

        # Raise error if index is not writable
        if not self.config.get("writable"):
            raise ReadOnlyError("Attempting to index a read-only index (writable != True)")

        if self.embeddings and self.documents:
            with self.lock:
                # Reset index
                self.indexes(False)

                # Build scoring index if term weighting is enabled
                if self.embeddings.isweighted():
                    self.embeddings.score(self.documents)

                # Build embeddings index
                self.embeddings.index(self.documents)

                # Save index if path available, otherwise this is an memory-only index
                if self.config.get("path"):
                    self.embeddings.save(self.config["path"], self.config.get("cloud"))

                # Reset document stream
                self.documents.close()
                self.documents = None

    def upsert(self):
        """
        Runs an embeddings upsert operation for previously batched documents.
        """

        # Raise error if index is not writable
        if not self.config.get("writable"):
            raise ReadOnlyError("Attempting to upsert a read-only index (writable != True)")

        if self.embeddings and self.documents:
            with self.lock:
                # Run upsert
                self.embeddings.upsert(self.documents)

                # Save index if path available, otherwise this is an memory-only index
                if self.config.get("path"):
                    self.embeddings.save(self.config["path"], self.config.get("cloud"))

                # Reset document stream
                self.documents.close()
                self.documents = None

    def delete(self, ids):
        """
        Deletes from an embeddings index. Returns list of ids deleted.

        Args:
            ids: list of ids to delete

        Returns:
            ids deleted
        """

        # Raise error if index is not writable
        if not self.config.get("writable"):
            raise ReadOnlyError("Attempting to delete from a read-only index (writable != True)")

        if self.embeddings:
            with self.lock:
                # Run delete operation
                deleted = self.embeddings.delete(ids)

                # Save index if path available, otherwise this is an memory-only index
                if self.config.get("path"):
                    self.embeddings.save(self.config["path"], self.config.get("cloud"))

                # Return deleted ids
                return deleted

        return None

    def reindex(self, config, function=None):
        """
        Recreates embeddings index using config. This method only works if document content storage is enabled.

        Args:
            config: new config
            function: optional function to prepare content for indexing
        """

        # Raise error if index is not writable
        if not self.config.get("writable"):
            raise ReadOnlyError("Attempting to reindex a read-only index (writable != True)")

        if self.embeddings:
            with self.lock:
                # Resolve function, if necessary
                function = self.function(function) if function and isinstance(function, str) else function

                # Reindex
                self.embeddings.reindex(config, function)

                # Save index if path available, otherwise this is an memory-only index
                if self.config.get("path"):
                    self.embeddings.save(self.config["path"], self.config.get("cloud"))

    def count(self):
        """
        Total number of elements in this embeddings index.

        Returns:
            number of elements in embeddings index
        """

        if self.embeddings:
            return self.embeddings.count()

        return None

    def similarity(self, query, texts):
        """
        Computes the similarity between query and list of text. Returns a list of
        {id: value, score: value} sorted by highest score, where id is the index
        in texts.

        Args:
            query: query text
            texts: list of text

        Returns:
            list of {id: value, score: value}
        """

        # Use similarity instance if available otherwise fall back to embeddings model
        if "similarity" in self.pipelines:
            return [{"id": uid, "score": float(score)} for uid, score in self.pipelines["similarity"](query, texts)]
        if self.embeddings:
            return [{"id": uid, "score": float(score)} for uid, score in self.embeddings.similarity(query, texts)]

        return None

    def batchsimilarity(self, queries, texts):
        """
        Computes the similarity between list of queries and list of text. Returns a list
        of {id: value, score: value} sorted by highest score per query, where id is the
        index in texts.

        Args:
            queries: queries text
            texts: list of text

        Returns:
            list of {id: value, score: value} per query
        """

        # Use similarity instance if available otherwise fall back to embeddings model
        if "similarity" in self.pipelines:
            return [[{"id": uid, "score": float(score)} for uid, score in r] for r in self.pipelines["similarity"](queries, texts)]
        if self.embeddings:
            return [[{"id": uid, "score": float(score)} for uid, score in r] for r in self.embeddings.batchsimilarity(queries, texts)]

        return None

    def explain(self, query, texts=None, limit=10):
        """
        Explains the importance of each input token in text for a query.

        Args:
            query: query text
            texts: optional list of text, otherwise runs search query
            limit: optional limit if texts is None

        Returns:
            list of dict per input text where a higher token scores represents higher importance relative to the query
        """

        if self.embeddings:
            with self.lock:
                return self.embeddings.explain(query, texts, limit)

        return None

    def batchexplain(self, queries, texts=None, limit=10):
        """
        Explains the importance of each input token in text for a list of queries.

        Args:
            query: queries text
            texts: optional list of text, otherwise runs search queries
            limit: optional limit if texts is None

        Returns:
            list of dict per input text per query where a higher token scores represents higher importance relative to the query
        """

        if self.embeddings:
            with self.lock:
                return self.embeddings.batchexplain(queries, texts, limit)

        return None

    def transform(self, text, category=None, index=None):
        """
        Transforms text into embeddings arrays.

        Args:
            text: input text
            category: category for instruction-based embeddings
            index: index name, if applicable

        Returns:
            embeddings array
        """

        if self.embeddings:
            return [float(x) for x in self.embeddings.transform(text, category, index)]

        return None

    def batchtransform(self, texts, category=None, index=None):
        """
        Transforms list of text into embeddings arrays.

        Args:
            texts: list of text
            category: category for instruction-based embeddings
            index: index name, if applicable

        Returns:
            embeddings arrays
        """

        if self.embeddings:
            return [[float(x) for x in result] for result in self.embeddings.batchtransform(texts, category, index)]

        return None

    def extract(self, queue, texts=None):
        """
        Extracts answers to input questions.

        Args:
            queue: list of {name: value, query: value, question: value, snippet: value}
            texts: optional list of text

        Returns:
            list of {name: value, answer: value}
        """

        if self.embeddings and "extractor" in self.pipelines:
            # Get extractor instance
            extractor = self.pipelines["extractor"]

            # Run extractor and return results as dicts
            return extractor(queue, texts)

        return None

    def label(self, text, labels):
        """
        Applies a zero shot classifier to text using a list of labels. Returns a list of
        {id: value, score: value} sorted by highest score, where id is the index in labels.

        Args:
            text: text|list
            labels: list of labels

        Returns:
            list of {id: value, score: value} per text element
        """

        if "labels" in self.pipelines:
            # Text is a string
            if isinstance(text, str):
                return [{"id": uid, "score": float(score)} for uid, score in self.pipelines["labels"](text, labels)]

            # Text is a list
            return [[{"id": uid, "score": float(score)} for uid, score in result] for result in self.pipelines["labels"](text, labels)]

        return None

    def pipeline(self, name, *args, **kwargs):
        """
        Generic pipeline execution method.

        Args:
            name: pipeline name
            args: pipeline positional arguments
            kwargs: pipeline keyword arguments

        Returns:
            pipeline results
        """

        # Backwards compatible with previous pipeline function arguments
        args = args[0] if args and len(args) == 1 and isinstance(args[0], tuple) else args

        if name in self.pipelines:
            return self.pipelines[name](*args, **kwargs)

        return None

    def workflow(self, name, elements):
        """
        Executes a workflow.

        Args:
            name: workflow name
            elements: elements to process

        Returns:
            processed elements
        """

        if hasattr(elements, "__len__") and hasattr(elements, "__getitem__"):
            # Convert to tuples and return as a list since input is sized
            elements = [tuple(element) if isinstance(element, list) else element for element in elements]
        else:
            # Convert to tuples and return as a generator since input is not sized
            elements = (tuple(element) if isinstance(element, list) else element for element in elements)

        # Execute workflow
        return self.workflows[name](elements)

    def agent(self, name, *args, **kwargs):
        """
        Executes an agent.

        Args:
            name: agent name
            args: agent positional arguments
            kwargs: agent keyword arguments
        """

        if name in self.agents:
            return self.agents[name](*args, **kwargs)

        return None

    def wait(self):
        """
        Closes threadpool and waits for completion.
        """

        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None


class ReadOnlyError(Exception):
    """
    Error raised when trying to modify a read-only index
    """

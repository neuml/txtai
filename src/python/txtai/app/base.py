"""
Application module
"""

import os

from multiprocessing.pool import ThreadPool
from threading import RLock

import yaml

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
            else:
                # Read yaml from string
                return yaml.safe_load(data)

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
        self.pipes()

        # Create workflows
        self.flows()

        # Create embeddings index
        self.indexes(loaddata)

    def __del__(self):
        """
        Close threadpool when this object is garbage collected.
        """

        if self.pool:
            self.pool.close()
            self.pool = None

    def pipes(self):
        """
        Initialize pipelines.
        """

        # Pipeline definitions
        self.pipelines = {}

        # Default pipelines
        pipelines = list(PipelineFactory.list().keys())

        # Add custom pipelines
        for key in self.config:
            if "." in key:
                pipelines.append(key)

        # Create pipelines
        for pipeline in pipelines:
            if pipeline in self.config:
                config = self.config[pipeline] if self.config[pipeline] else {}

                # Custom pipeline parameters
                if pipeline == "extractor":
                    config["similarity"] = self.embeddings
                elif pipeline == "similarity" and "path" not in config and "labels" in self.pipelines:
                    config["model"] = self.pipelines["labels"]

                self.pipelines[pipeline] = PipelineFactory.create(config, pipeline)

    def flows(self):
        """
        Initialize workflows.
        """

        # Workflow definitions
        self.workflows = {}

        # Create workflows
        if "workflow" in self.config:
            for workflow, config in self.config["workflow"].items():
                # Create copy of config
                config = config.copy()

                # Resolve callable functions
                config["tasks"] = [self.resolve(task) for task in config["tasks"]]

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

    def indexes(self, loaddata):
        """
        Initialize an embeddings index.

        Args:
            loaddata: If True (default), load existing index data, if available. Otherwise, only load models.
        """

        # Resolve functions in embeddings config
        config = self.config.get("embeddings") if self.config else None
        if config:
            # Create copy of config
            config = config.copy()

            if "functions" in config:
                # Resolve callable functions
                functions = []
                for fn in config["functions"]:
                    if isinstance(fn, dict):
                        fn["function"] = self.function(fn["function"])
                    else:
                        fn = self.function(fn)
                    functions.append(fn)

                config["functions"] = functions

            if "transform" in config:
                # Resolve transform function
                config["transform"] = self.function(config["transform"])

        # Local embeddings index
        if loaddata and self.config.get("path") and Embeddings().exists(self.config["path"], self.config.get("cloud")):
            # Load existing index if available
            self.embeddings = Embeddings()
            self.embeddings.load(self.config["path"], self.config.get("cloud"))
        elif config:
            # Initialize empty embeddings
            self.embeddings = Embeddings(config)

    def resolve(self, task):
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

    def function(self, function):
        """
        Get a handle to a callable function.

        Args:
            function: function name

        Returns:
            resolved function
        """

        if function in self.pipelines:
            return self.pipelines[function]

        # Attempt to resolve action as a callable function
        return PipelineFactory.create({}, function)

    def search(self, query, limit=10):
        """
        Finds documents in the embeddings model most similar to the input query. Returns
        a list of {id: value, score: value} sorted by highest score, where id is the
        document id in the embeddings model.

        Downstream applications can override this method to provide enriched search results.

        Args:
            query: query text
            limit: maximum results, used if request is None

        Returns:
            list of {id: value, score: value}
        """

        if self.embeddings:
            # Unpack (id, score) tuple, if necessary. Otherwise, results are dictionaries.
            return [{"id": r[0], "score": float(r[1])} if isinstance(r, tuple) else r for r in self.embeddings.search(query, limit)]

        return None

    def batchsearch(self, queries, limit=10):
        """
        Finds documents in the embeddings model most similar to the input queries. Returns
        a list of {id: value, score: value} sorted by highest score per query, where id is
        the document id in the embeddings model.

        Args:
            queries: queries text
            limit: maximum results

        Returns:
            list of {id: value, score: value} per query
        """

        if self.embeddings:
            with self.lock:
                search = self.embeddings.batchsearch(queries, limit)

            results = []
            for result in search:
                # Unpack (id, score) tuple, if necessary. Otherwise, results are dictionaries.
                results.append([{"id": r[0], "score": float(r[1])} if isinstance(r, tuple) else r for r in result])
            return results

        return None

    def add(self, documents):
        """
        Adds a batch of documents for indexing.

        Downstream applications can override this method to also store full documents in an external system.

        Args:
            documents: list of {id: value, text: value}

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

                batch = []
                index = self.count() + len(self.documents)
                for document in documents:
                    if isinstance(document, dict):
                        # Pass dictionary, the embeddings instance handles parsing out the "text" field
                        document = (document["id"], document, None)
                    elif isinstance(document, str):
                        # Add id via autosequence
                        document = (index, document, None)
                        index += 1
                    elif isinstance(document, tuple) and len(document) < 3:
                        # Copy partial tuple
                        document = (document[0], document[1], None)

                    # Add document tuple (id, text, element)
                    batch.append(document)

                # Add batch
                self.documents.add(batch)

        # Return unmodified input documents
        return documents

    def index(self):
        """
        Builds an embeddings index for previously batched documents.
        """

        # Raise error if index is not writable
        if not self.config.get("writable"):
            raise ReadOnlyError("Attempting to index a read-only index (writable != True)")

        if self.embeddings and self.documents:
            with self.lock:
                # Build scoring index if scoring method provided
                if self.embeddings.scoring:
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
                    self.embeddings.save(self.config["path"])

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
                return self.embeddings.delete(ids)

        return None

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

    def transform(self, text):
        """
        Transforms text into embeddings arrays.

        Args:
            text: input text

        Returns:
            embeddings array
        """

        if self.embeddings:
            return [float(x) for x in self.embeddings.transform((None, text, None))]

        return None

    def batchtransform(self, texts):
        """
        Transforms list of text into embeddings arrays.

        Args:
            texts: list of text

        Returns:
            embeddings arrays
        """

        if self.embeddings:
            documents = [(None, text, None) for text in texts]
            return [[float(x) for x in result] for result in self.embeddings.batchtransform(documents)]

        return None

    def extract(self, queue, texts):
        """
        Extracts answers to input questions.

        Args:
            queue: list of {name: value, query: value, question: value, snippet: value}
            texts: list of text

        Returns:
            list of {name: value, answer: value}
        """

        if self.embeddings and "extractor" in self.pipelines:
            # Set similarity function
            extractor = self.pipelines["extractor"]
            if not extractor.similarity:
                extractor.similarity = self.embeddings

            # Convert queue to tuples
            queue = [(x["name"], x["query"], x.get("question"), x.get("snippet")) for x in queue]
            return [{"name": name, "answer": answer} for name, answer in extractor(queue, texts)]

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

    def pipeline(self, name, args):
        """
        Generic pipeline execution method.

        Args:
            name: pipeline name
            args: pipeline arguments
        """

        if name in self.pipelines:
            return self.pipelines[name](*args)

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

        # Convert lists to tuples
        elements = [tuple(element) if isinstance(element, list) else element for element in elements]

        # Execute workflow
        return self.workflows[name](elements)

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

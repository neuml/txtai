"""
API module
"""

from .cluster import Cluster

from ..embeddings import Documents, Embeddings
from ..pipeline import PipelineFactory
from ..workflow import WorkflowFactory


class API:
    """
    Base API template. Downstream applications can extend this base template to add/modify functionality.
    """

    def __init__(self, config):
        """
        Creates an embeddings index instance that is called by FastAPI.

        Args:
            config: index configuration
        """

        # Initialize member variables
        self.config, self.documents, self.embeddings, self.cluster = config, None, None, None

        # Local embeddings index
        if self.config.get("path") and Embeddings().exists(self.config["path"]):
            # Load existing index if available
            self.embeddings = Embeddings()
            self.embeddings.load(self.config["path"])
        elif self.config.get("embeddings"):
            # Initialize empty embeddings
            self.embeddings = Embeddings(self.config["embeddings"])

        # Embeddings cluster
        if self.config.get("cluster"):
            self.cluster = Cluster(self.config["cluster"])

        # Create pipelines
        self.pipes()

        # Create workflows
        self.flows()

    def pipes(self):
        """
        Initialize pipelines.
        """

        # Pipeline definitions
        self.pipelines = {}

        # Default pipelines
        pipelines = ["extractor", "labels", "segmentation", "similarity", "summary", "textractor", "transcription", "translation"]

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
                    config["embeddings"] = self.embeddings
                elif pipeline == "similarity" and "path" not in config and "labels" in self.pipelines:
                    config["model"] = self.pipelines["labels"]

                self.pipelines[pipeline] = PipelineFactory.create(config, pipeline)

    def flows(self):
        """
        Initialize workflows.
        """

        self.workflows = {}

        # Create workflows
        if "workflow" in self.config:
            for workflow, config in self.config["workflow"].items():
                for task in config["tasks"]:
                    # Resolve pipeline action
                    if "action" in task:
                        task["action"] = self.pipelines[task["action"]]

                self.workflows[workflow] = WorkflowFactory.create(config)

    def limit(self, limit):
        """
        Parses the number of results to return from the request. Allows range of 1-250, with a default of 10.

        Args:
            limit: limit parameter

        Returns:
            bounded limit
        """

        # Return between 1 and 250 results, defaults to 10
        return max(1, min(250, int(limit) if limit else 10))

    def search(self, query, request):
        """
        Finds documents in the embeddings model most similar to the input query. Returns
        a list of {id: value, score: value} sorted by highest score, where id is the
        document id in the embeddings model.

        Downstream applications can override this method to provide enriched search results.

        Args:
            query: query text
            request: FastAPI request

        Returns:
            list of {id: value, score: value}
        """

        limit = self.limit(request.query_params.get("limit") if request else None)

        if self.cluster:
            return self.cluster.search(query, limit)
        if self.embeddings:
            return [{"id": uid, "score": float(score)} for uid, score in self.embeddings.search(query, limit)]

        return None

    def batchsearch(self, queries, limit):
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

        if self.cluster:
            return self.cluster.batchsearch(queries, self.limit(limit))
        if self.embeddings:
            results = self.embeddings.batchsearch(queries, self.limit(limit))
            return [[{"id": uid, "score": float(score)} for uid, score in result] for result in results]

        return None

    def add(self, documents):
        """
        Adds a batch of documents for indexing.

        Downstream applications can override this method to also store full documents in an external system.

        Args:
            documents: list of {id: value, text: value}
        """

        if self.cluster:
            self.cluster.add(documents)
        elif self.embeddings and self.config.get("writable"):
            # Only add batch if index is marked writable
            # Create documents file if not already open
            if not self.documents:
                self.documents = Documents()

            # Add batch
            self.documents.add([(document["id"], document["text"], None) for document in documents])

    def index(self):
        """
        Builds an embeddings index for previously batched documents.
        """

        if self.cluster:
            self.cluster.index()
        elif self.embeddings and self.config.get("writable") and self.documents:
            # Build scoring index if scoring method provided
            if self.config.get("scoring"):
                self.embeddings.score(self.documents)

            # Build embeddings index
            self.embeddings.index(self.documents)

            # Save index if path available, otherwise this is an memory-only index
            if self.config.get("path"):
                self.embeddings.save(self.config["path"])

            # Reset document stream
            self.documents.close()
            self.documents = None

    def upsert(self):
        """
        Runs an embeddings upsert operation for previously batched documents.
        """

        if self.cluster:
            self.cluster.upsert()
        elif self.embeddings and self.config.get("writable") and self.documents:
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

        if self.cluster:
            return self.cluster.delete(ids)
        if self.embeddings and self.config.get("writable"):
            return self.embeddings.delete(ids)

        return None

    def count(self):
        """
        Total number of elements in this embeddings index.

        Returns:
            number of elements in embeddings index
        """

        if self.cluster:
            return self.cluster.count()
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
            # Convert queue to tuples
            queue = [(x["name"], x["query"], x.get("question"), x.get("snippet")) for x in queue]
            return [{"name": name, "answer": answer} for name, answer in self.pipelines["extractor"](queue, texts)]

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

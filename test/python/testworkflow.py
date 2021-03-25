"""
Workflow module tests
"""

import unittest

from txtai.embeddings import Documents, Embeddings
from txtai.pipeline import Summary, Translation, Textractor
from txtai.workflow import Workflow, Task, FileTask, ImageTask, WorkflowTask

# pylint: disable = C0411
from utils import Utils


class TestWorkFlow(unittest.TestCase):
    """
    Workflow tests
    """

    def testBaseWorkflow(self):
        """
        Tests a basic workflow
        """

        translate = Translation()

        # Workflow that translate text to Spanish
        workflow = Workflow([Task(lambda x: translate(x, "es"))])

        results = list(workflow(["The sky is blue", "Forest through the trees"]))

        self.assertEqual(len(results), 2)

    def testComplexWorkflow(self):
        """
        Tests a complex workflow
        """

        textractor = Textractor(paragraphs=True, minlength=150, join=True)
        summary = Summary()
        translate = Translation()

        embeddings = Embeddings({"method": "transformers", "path": "sentence-transformers/paraphrase-xlm-r-multilingual-v1"})
        documents = Documents()

        def index(x):
            documents.add(x)
            return x

        # Extract text and summarize articles
        articles = Workflow([FileTask(textractor), Task(lambda x: summary(x, maxlength=15))])

        # Complex workflow that handles text extraction and audio transcription
        # Results are translated to Spanish and loaded into an embeddings index
        tasks = [WorkflowTask(articles, r".\.pdf$"), Task(lambda x: translate(x, "es")), Task(index, unpack=False)]

        data = ["article.pdf", "Workflows can process audio files, documents and snippets"]

        # Convert file paths to data tuples
        data = [(x, "file:///%s/%s" % (Utils.PATH, element), None) for x, element in enumerate(data)]

        # Execute workflow, discard results as they are streamed
        workflow = Workflow(tasks)
        for _ in workflow(data):
            pass

        # Build the embeddings index
        embeddings.index(documents)

        # Cleanup temporary storage
        documents.close()

        # Run search and validate result
        index, _ = embeddings.search("buscar texto", 1)[0]
        self.assertEqual(index, 0)

    def testImageWorkflow(self):
        """
        Tests an image task
        """

        workflow = Workflow([ImageTask()])

        results = list(workflow(["file://" + Utils.PATH + "/books.jpg"]))

        self.assertEqual(results[0].size, (1024, 682))

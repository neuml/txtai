"""
Workflow module tests
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from txtai.api import API
from txtai.embeddings import Documents, Embeddings
from txtai.pipeline import Nop, Segmentation, Summary, Translation, Textractor
from txtai.workflow import Workflow, Task, FileTask, ImageTask, RetrieveTask, StorageTask, WorkflowTask

# pylint: disable = C0411
from utils import Utils


class TestWorkflow(unittest.TestCase):
    """
    Workflow tests.
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

    def testChainWorkflow(self):
        """
        Tests a chain of workflows
        """

        workflow1 = Workflow([Task(lambda x: [y * 2 for y in x])])
        workflow2 = Workflow([Task(lambda x: [y - 1 for y in x])], batch=4)

        results = list(workflow2(workflow1([1, 2, 4, 8, 16, 32])))
        self.assertEqual(results, [1, 3, 7, 15, 31, 63])

    def testComplexWorkflow(self):
        """
        Tests a complex workflow
        """

        textractor = Textractor(paragraphs=True, minlength=150, join=True)
        summary = Summary("t5-small")

        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
        documents = Documents()

        def index(x):
            documents.add(x)
            return x

        # Extract text and summarize articles
        articles = Workflow([FileTask(textractor), Task(lambda x: summary(x, maxlength=15))])

        # Complex workflow that extracts text, runs summarization then loads into an embeddings index
        tasks = [WorkflowTask(articles, r".\.pdf$"), Task(index, unpack=False)]

        data = ["file://" + Utils.PATH + "/article.pdf", "Workflows can process audio files, documents and snippets"]

        # Convert file paths to data tuples
        data = [(x, element, None) for x, element in enumerate(data)]

        # Execute workflow, discard results as they are streamed
        workflow = Workflow(tasks)
        data = list(workflow(data))

        # Build the embeddings index
        embeddings.index(documents)

        # Cleanup temporary storage
        documents.close()

        # Run search and validate result
        index, _ = embeddings.search("search text", 1)[0]
        self.assertEqual(index, 0)
        self.assertEqual(data[0][1], "txtai builds an AI-powered index over sections")

    def testConcurrentWorkflow(self):
        """
        Tests running concurrent task actions
        """

        nop = Nop()

        workflow = Workflow([Task([nop, nop], concurrency="thread")])
        results = list(workflow([2, 4]))
        self.assertEqual(results, [(2, 2), (4, 4)])

        workflow = Workflow([Task([nop, nop], concurrency="process")])
        results = list(workflow([2, 4]))
        self.assertEqual(results, [(2, 2), (4, 4)])

        workflow = Workflow([Task([nop, nop], concurrency="unknown")])
        results = list(workflow([2, 4]))
        self.assertEqual(results, [(2, 2), (4, 4)])

    def testExtractWorkflow(self):
        """
        Tests column extraction tasks
        """

        workflow = Workflow([Task(lambda x: x, unpack=False, column=0)], batch=1)

        results = list(workflow([(0, 1)]))
        self.assertEqual(results[0], 0)

        results = list(workflow([(0, (1, 2), None)]))
        self.assertEqual(results[0], (0, 1, None))

        results = list(workflow([1]))
        self.assertEqual(results[0], 1)

    def testImageWorkflow(self):
        """
        Tests an image task
        """

        workflow = Workflow([ImageTask()])

        results = list(workflow([Utils.PATH + "/books.jpg"]))

        self.assertEqual(results[0].size, (1024, 682))

    def testInvalidWorkflow(self):
        """
        Tests task with invalid parameters
        """

        with self.assertRaises(TypeError):
            Task(invalid=True)

    def testMergeWorkflow(self):
        """
        Tests merge tasks
        """

        task = Task([lambda x: [pow(y, 2) for y in x], lambda x: [pow(y, 3) for y in x]], merge="hstack")

        # Test hstack (column-wise) merge
        workflow = Workflow([task])
        results = list(workflow([2, 4]))
        self.assertEqual(results, [(4, 8), (16, 64)])

        # Test vstack (row-wise) merge
        task.merge = "vstack"
        results = list(workflow([2, 4]))
        self.assertEqual(results, [4, 8, 16, 64])

        # Test concat (values joined into single string) merge
        task.merge = "concat"
        results = list(workflow([2, 4]))
        self.assertEqual(results, ["4. 8", "16. 64"])

        # Test no merge
        task.merge = None
        results = list(workflow([2, 4, 6]))
        self.assertEqual(results, [[4, 16, 36], [8, 64, 216]])

        # Test generated (id, data, tag) tuples are properly returned
        workflow = Workflow([Task(lambda x: [(0, y, None) for y in x])])
        results = list(workflow([(1, "text", "tags")]))
        self.assertEqual(results[0], (0, "text", None))

    def testMergeUnbalancedWorkflow(self):
        """
        Test merge tasks with unbalanced outputs (i.e. one action produce more output than another for same input).
        """

        nop = Nop()
        segment1 = Segmentation(sentences=True)

        task = Task([nop, segment1])

        # Test hstack
        workflow = Workflow([task])
        results = list(workflow(["This is a test sentence. And another sentence to split."]))
        self.assertEqual(
            results, [("This is a test sentence. And another sentence to split.", ["This is a test sentence.", "And another sentence to split."])]
        )

        # Test vstack
        task.merge = "vstack"
        workflow = Workflow([task])
        results = list(workflow(["This is a test sentence. And another sentence to split."]))
        self.assertEqual(
            results, ["This is a test sentence. And another sentence to split.", "This is a test sentence.", "And another sentence to split."]
        )

    def testNumpyWorkflow(self):
        """
        Tests a numpy workflow
        """

        task = Task([lambda x: np.power(x, 2), lambda x: np.power(x, 3)], merge="hstack")

        # Test hstack (column-wise) merge
        workflow = Workflow([task])
        results = list(workflow(np.array([2, 4])))
        self.assertTrue(np.array_equal(np.array(results), np.array([[4, 8], [16, 64]])))

        # Test vstack (row-wise) merge
        task.merge = "vstack"
        results = list(workflow(np.array([2, 4])))
        self.assertEqual(results, [4, 8, 16, 64])

        # Test no merge
        task.merge = None
        results = list(workflow(np.array([2, 4, 6])))
        self.assertTrue(np.array_equal(np.array(results), np.array([[4, 16, 36], [8, 64, 216]])))

    def testRetrieveWorkflow(self):
        """
        Tests a retrieve task
        """

        # Test retrieve with generated temporary directory
        workflow = Workflow([RetrieveTask()])
        results = list(workflow(["file://" + Utils.PATH + "/books.jpg"]))
        self.assertTrue(results[0].endswith("books.jpg"))

        # Test retrieve with specified temporary directory
        workflow = Workflow([RetrieveTask(directory=os.path.join(tempfile.gettempdir(), "retrieve"))])
        results = list(workflow(["file://" + Utils.PATH + "/books.jpg"]))
        self.assertTrue(results[0].endswith("books.jpg"))

    def testStorageWorkflow(self):
        """
        Tests a storage task
        """

        workflow = Workflow([StorageTask()])

        results = list(workflow(["local://" + Utils.PATH, "test string"]))

        self.assertEqual(len(results), 19)

    def testTensorTransformWorkflow(self):
        """
        Tests a tensor workflow with list transformations
        """

        # Test one-one list transformation
        task = Task(lambda x: x.tolist())
        workflow = Workflow([task])
        results = list(workflow(np.array([2])))
        self.assertEqual(results, [2])

        # Test one-many list transformation
        task = Task(lambda x: [x.tolist() * 2])
        workflow = Workflow([task])
        results = list(workflow(np.array([2])))
        self.assertEqual(results, [2, 2])

    def testTorchWorkflow(self):
        """
        Tests a torch workflow
        """

        # pylint: disable=E1101,E1102
        task = Task([lambda x: torch.pow(x, 2), lambda x: torch.pow(x, 3)], merge="hstack")

        # Test hstack (column-wise) merge
        workflow = Workflow([task])
        results = np.array([x.numpy() for x in workflow(torch.tensor([2, 4]))])
        self.assertTrue(np.array_equal(results, np.array([[4, 8], [16, 64]])))

        # Test vstack (row-wise) merge
        task.merge = "vstack"
        results = list(workflow(torch.tensor([2, 4])))
        self.assertEqual(results, [4, 8, 16, 64])

        # Test no merge
        task.merge = None
        results = np.array([x.numpy() for x in workflow(torch.tensor([2, 4, 6]))])
        self.assertTrue(np.array_equal(np.array(results), np.array([[4, 16, 36], [8, 64, 216]])))

    def testYamlWorkflow(self):
        """
        Tests reading a YAML workflow in Python.
        """

        # Read from string
        config = """
        # Embeddings index
        writable: true
        embeddings:
            scoring: bm25
            path: google/bert_uncased_L-2_H-128_A-2

        # Text segmentation
        segmentation:
            sentences: true

        # Workflow definitions
        workflow:
            segment:
                tasks:
                    - action: segmentation
                    - action: index
        """

        app = API(config)
        self.assertEqual(
            list(app.workflow("segment", ["This is a test sentence. And another sentence to split."])),
            ["This is a test sentence.", "And another sentence to split."],
        )

        # Read from file
        path = os.path.join(tempfile.gettempdir(), "workflow.yml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(config)

        app = API(path)
        self.assertEqual(
            list(app.workflow("segment", ["This is a test sentence. And another sentence to split."])),
            ["This is a test sentence.", "And another sentence to split."],
        )

        # Read from YAML object
        app = API(API.read(config))
        self.assertEqual(
            list(app.workflow("segment", ["This is a test sentence. And another sentence to split."])),
            ["This is a test sentence.", "And another sentence to split."],
        )

    def testYamlError(self):
        """
        Tests reading a YAML workflow with errors.
        """

        # Read from string
        config = """
        # Workflow definitions
        workflow:
            error:
                tasks:
                    - action: error
        """

        with self.assertRaises(KeyError):
            API(config)

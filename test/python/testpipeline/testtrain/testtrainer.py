"""
Trainer module tests
"""

import os
import unittest
import tempfile

from unittest.mock import patch

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from txtai.data import Data
from txtai.pipeline import HFTrainer, Labels, Questions, Sequences


class TestTrainer(unittest.TestCase):
    """
    Trainer tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create default datasets.
        """

        cls.data = [{"text": "Dogs", "label": 0}, {"text": "dog", "label": 0}, {"text": "Cats", "label": 1}, {"text": "cat", "label": 1}] * 100

    def testBasic(self):
        """
        Test training a model with basic parameters
        """

        trainer = HFTrainer()
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", self.data)

        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels("cat")[0][0], 1)

    def testCLM(self):
        """
        Test training a model with causal language modeling
        """

        trainer = HFTrainer()
        model, _ = trainer("hf-internal-testing/tiny-random-gpt2", self.data, maxlength=16, task="language-generation")

        # Test model completed successfully
        self.assertIsNotNone(model)

    def testCustom(self):
        """
        Test training a model with custom parameters
        """

        # pylint: disable=E1120
        model = AutoModelForSequenceClassification.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

        trainer = HFTrainer()
        model, tokenizer = trainer(
            (model, tokenizer),
            self.data,
            self.data,
            columns=("text", "label"),
            do_eval=True,
            output_dir=os.path.join(tempfile.gettempdir(), "trainer"),
        )

        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels("cat")[0][0], 1)

    def testDataFrame(self):
        """
        Test training a model with a mock pandas DataFrame
        """

        class TestDataFrame:
            """
            Test DataFrame
            """

            def __init__(self, data):
                # Get list of columns
                self.columns = list(data[0].keys())

                # Build columnar data view
                self.data = {}
                for column in self.columns:
                    self.data[column] = Values([row[column] for row in data])

            def __getitem__(self, column):
                return self.data[column]

        class Values:
            """
            Test values list
            """

            def __init__(self, values):
                self.values = list(values)

            def __getitem__(self, index):
                return self.values[index]

            def unique(self):
                """
                Returns a list of unique values.

                Returns:
                    unique list of values
                """

                return set(self.values)

        # Mock DataFrame
        df = TestDataFrame(self.data)

        trainer = HFTrainer()
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", df)

        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels("cat")[0][0], 1)

    def testDataset(self):
        """
        Test training a model with a mock Hugging Face Dataset
        """

        class TestDataset(torch.utils.data.Dataset):
            """
            Test Dataset
            """

            def __init__(self, data):
                self.data = data
                self.unique = lambda _: [0, 1]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return self.data[index]

            def column_names(self):
                """
                Returns column names for this dataset

                Returns:
                    list of columns
                """

                return ["text", "label"]

            # pylint: disable=W0613
            def map(self, fn, batched, num_proc, remove_columns):
                """
                Map each dataset row using fn.

                Args:
                    fn: function
                    batched: batch records

                Returns:
                    updated Dataset
                """

                self.data = [fn(x) for x in self.data]
                return self

        ds = TestDataset(self.data)

        trainer = HFTrainer()
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", ds)

        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels("cat")[0][0], 1)

    def testEmpty(self):
        """
        Test an empty training data object
        """

        self.assertIsNone(Data(None, None, None).process(None))

    def testMLM(self):
        """
        Test training a model with masked language modeling.
        """

        trainer = HFTrainer()
        model, _ = trainer("hf-internal-testing/tiny-random-bert", self.data, task="language-modeling")

        # Test model completed successfully
        self.assertIsNotNone(model)

    def testMultiLabel(self):
        """
        Test training model with labels provided as a list
        """

        data = []
        for x in self.data:
            data.append({"text": x["text"], "label": [0.0, 1.0] if x["label"] else [1.0, 0.0]})

        trainer = HFTrainer()
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", data)

        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels("cat")[0][0], 1)

    @patch("importlib.util.find_spec")
    def testPEFT(self, spec):
        """
        Test training a model with causal language modeling and PEFT
        """

        # Disable triton
        spec.return_value = None

        trainer = HFTrainer()
        model, _ = trainer(
            "hf-internal-testing/tiny-random-gpt2",
            self.data,
            maxlength=16,
            task="language-generation",
            quantize=True,
            lora=True,
        )

        # Test model completed successfully
        self.assertIsNotNone(model)

    def testQA(self):
        """
        Test training a QA model
        """

        # Training data
        data = [
            {"question": "What ingredient?", "context": "1 can whole tomatoes", "answers": "tomatoes"},
            {"question": "What ingredient?", "context": "Crush 1 tomato", "answers": "tomato"},
            {"question": "What ingredient?", "context": "1 yellow onion", "answers": "onion"},
            {"question": "What ingredient?", "context": "Unwrap 2 red onions", "answers": "onions"},
            {"question": "What ingredient?", "context": "1 red pepper", "answers": "pepper"},
            {"question": "What ingredient?", "context": "Clean 3 red peppers", "answers": "peppers"},
            {"question": "What ingredient?", "context": "1 clove garlic", "answers": "garlic"},
            {"question": "What ingredient?", "context": "Unwrap 3 cloves of garlic", "answers": "garlic"},
            {"question": "What ingredient?", "context": "3 pieces of ginger", "answers": "ginger"},
            {"question": "What ingredient?", "context": "Peel 1 orange", "answers": "orange"},
            {"question": "What ingredient?", "context": "1/2 lb beef", "answers": "beef"},
            {"question": "What ingredient?", "context": "Roast 3 lbs of beef", "answers": "beef"},
            {"question": "What ingredient?", "context": "1 pack of chicken", "answers": "chicken"},
            {"question": "What ingredient?", "context": "Forest through the trees", "answers": None},
        ]

        trainer = HFTrainer()
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", data, data, task="question-answering", num_train_epochs=40)

        questions = Questions((model, tokenizer), gpu=True)
        self.assertEqual(questions(["What ingredient?"], ["Peel 1 onion"])[0], "onion")

    def testRegression(self):
        """
        Test training a model with a regression (continuous) output
        """

        data = []
        for x in self.data:
            data.append({"text": x["text"], "label": x["label"] + 0.1})

        trainer = HFTrainer()
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", data)

        labels = Labels((model, tokenizer), dynamic=False)

        # Regression tasks return a single entry with the regression output
        self.assertGreater(labels("cat")[0][1], 0.5)

    def testRTD(self):
        """
        Test training a language model with replaced token detection
        """

        # Save directory
        output = os.path.join(tempfile.gettempdir(), "trainer.rtd")

        trainer = HFTrainer()
        model, _ = trainer("hf-internal-testing/tiny-random-electra", self.data, task="token-detection", save_safetensors=False, output_dir=output)

        # Test model completed successfully
        self.assertIsNotNone(model)

        # Test output directories exist
        self.assertTrue(os.path.exists(os.path.join(output, "generator")))
        self.assertTrue(os.path.exists(os.path.join(output, "discriminator")))

    def testSeqSeq(self):
        """
        Test training a sequence-sequence model
        """

        data = [
            {"source": "Running again", "target": "Sleeping again"},
            {"source": "Run", "target": "Sleep"},
            {"source": "running", "target": "sleeping"},
        ]

        trainer = HFTrainer()
        model, tokenizer = trainer("t5-small", data, task="sequence-sequence", prefix="translate Run to Sleep: ", learning_rate=1e-3)

        # Run run-sleep translation
        sequences = Sequences((model, tokenizer))
        result = sequences("translate Run to Sleep: run")
        self.assertEqual(result.lower(), "sleep")

"""
Trainer module tests
"""

# pylint: disable=too-many-public-methods

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from txtai.data import Data
from txtai.models import Lemur, Models, PoolingFactory
from txtai.pipeline import HFTrainer, Labels, LemurTrainer, Questions, Sequences


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

        # Test default parameters
        model, _ = trainer("hf-internal-testing/tiny-random-gpt2", self.data, maxlength=16, task="language-generation")
        self.assertIsNotNone(model)

        # Test pack merging
        model, _ = trainer("hf-internal-testing/tiny-random-gpt2", self.data, maxlength=16, task="language-generation", merge="pack")
        self.assertIsNotNone(model)

        # Test no merging
        model, _ = trainer("hf-internal-testing/tiny-random-gpt2", self.data, maxlength=16, task="language-generation", merge=None)
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
            def map(self, fn, batched, batch_size, num_proc, remove_columns):
                """
                Map each dataset row using fn.

                Args:
                    fn: function
                    args: additional keyword args

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

    def testKD(self):
        """
        Test knowledge distillation
        """

        # Base model
        trainer = HFTrainer()
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", self.data)

        # Train with knowledge distillation
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", self.data, teacher=(model, tokenizer))

        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels("cat")[0][0], 1)

    def testLemurTrainer(self):
        """
        Test LEMUR trainer artifact round-trip and seeded determinism
        """

        model = "neuml/colbert-bert-tiny"
        corpus = [
            "alpha beta gamma",
            "beta gamma delta",
            "retrieval with token vectors",
            "fixed dimensional document weights",
            "learned maximum similarity",
            "deterministic trainer artifacts",
        ] * 2
        settings = {
            "gpu": False,
            "epochs": 0,
            "finalhiddendim": 128,
            "trainsubsetsize": 12,
            "learnsubsetsize": 128,
            "olssamplesize": 64,
            "seed": 42,
        }

        raw = PoolingFactory.create(
            {
                "path": model,
                "device": Models.deviceid(False),
                "modelargs": {"muvera": None, "lemur": None},
            }
        )
        documents = [raw.encode([text], batch=1, category="data")[0] for text in corpus[:3]]
        queries = [raw.encode([text], batch=1, category="query")[0] for text in corpus[:2]]

        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
            tempfile.TemporaryDirectory() as third,
        ):
            trained = LemurTrainer()(model, corpus, first, **settings)
            reloaded = Lemur(first)
            explicitquery = LemurTrainer()(model, corpus, second, learn=corpus, learncategory="query", **settings)
            datalearn = LemurTrainer()(model, corpus, third, learn=corpus, learncategory="data", **settings)
            queryreloaded = Lemur(second)

            trainedqueries = torch.from_numpy(trained(queries, "query"))
            traineddocuments = torch.from_numpy(trained(documents, "data"))
            datadocuments = torch.from_numpy(datalearn(documents, "data"))

            # Float32 encoder and SVD kernels can vary across Torch/BLAS builds.
            self.assertTrue(torch.allclose(trainedqueries, torch.from_numpy(reloaded(queries, "query")), rtol=1e-5, atol=1e-6))
            self.assertTrue(torch.allclose(traineddocuments, torch.from_numpy(reloaded(documents, "data")), rtol=1e-5, atol=1e-6))
            np.testing.assert_allclose(trainedqueries.numpy(), explicitquery(queries, "query"), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(traineddocuments.numpy(), explicitquery(documents, "data"), rtol=1e-5, atol=1e-6)
            self.assertFalse(torch.allclose(trained.sample, datalearn.sample, rtol=1e-5, atol=1e-6))
            self.assertFalse(torch.allclose(traineddocuments, datadocuments, rtol=1e-5, atol=1e-6))
            self.assertTrue(torch.allclose(traineddocuments, torch.from_numpy(queryreloaded(documents, "data")), rtol=1e-5, atol=1e-6))

    def testLemurTrainerCorpusSubset(self):
        """
        Test LEMUR samples corpus texts before deterministic encoding
        """

        class Pooling:
            """
            Records raw text encoding calls.
            """

            def __init__(self):
                self.calls = []

            def encode(self, texts, batch, category):
                """
                Records and returns a synthetic token vector.
                """

                self.calls.append((texts[0], category, batch))
                return [torch.ones((1, 2))]

        corpus = [f"document {index}" for index in range(12)]
        runs = []
        for _ in range(2):
            pooling = Pooling()
            with (
                patch("txtai.pipeline.train.lemur.PoolingFactory.create", return_value=pooling),
                patch.object(LemurTrainer, "fit", autospec=True, return_value=None),
            ):
                LemurTrainer()("model", corpus, "output", gpu=False, epochs=0, corpussubsetsize=4, seed=7)
            runs.append(pooling.calls)

        self.assertEqual(runs[0], runs[1])
        data = [text for text, category, _ in runs[0] if category == "data"]
        learn = [text for text, category, _ in runs[0] if category == "query"]
        self.assertEqual(len(data), 4)
        self.assertEqual(data, learn)

        pooling = Pooling()
        with (
            patch("txtai.pipeline.train.lemur.PoolingFactory.create", return_value=pooling),
            patch.object(LemurTrainer, "fit", autospec=True, return_value=None),
        ):
            LemurTrainer()("model", corpus, "output", gpu=False, epochs=0)
        self.assertEqual(len([call for call in pooling.calls if call[1] == "data"]), len(corpus))
        self.assertEqual(len([call for call in pooling.calls if call[1] == "query"]), len(corpus))

        with self.assertRaisesRegex(ValueError, "corpussubsetsize must be a positive integer"):
            LemurTrainer()("model", corpus, "output", gpu=False, epochs=0, corpussubsetsize=0)

    def testLemurTrainerValidation(self):
        """
        Test LEMUR trainer validates inputs before loading a model
        """

        tests = [
            ([], {"epochs": 0}, "data must contain at least one corpus text"),
            (["text"], {"epochs": 0, "learncategory": "invalid"}, "learncategory must be data or query"),
            (["text"], {}, "epochs must be set explicitly"),
            (["text"], {"epochs": 0, "learn": []}, "learn must contain at least one text"),
        ]

        with patch("txtai.pipeline.train.lemur.PoolingFactory.create") as create:
            for data, settings, message in tests:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        LemurTrainer()("model", data, "output", gpu=False, **settings)

            create.assert_not_called()

    def testLemurRoundTrip(self):
        """
        Test an ordinary LEMUR artifact save/load round-trip
        """

        random = np.random.default_rng(42)
        documents = [random.normal(size=(5, 6)).astype(np.float32) for _ in range(8)]
        queries = [random.normal(size=(3, 6)).astype(np.float32) for _ in range(2)]

        with tempfile.TemporaryDirectory() as output:
            fitted = LemurTrainer().fit(
                documents,
                output=output,
                epochs=0,
                finalhiddendim=10,
                trainsubsetsize=8,
                learnsubsetsize=40,
                olssamplesize=24,
                seed=42,
            )
            self.assertEqual(set(os.listdir(output)), {"config.json", "model.safetensors"})
            loaded = Lemur(output)

            # Float32 feature and SVD kernels can vary across Torch/BLAS builds.
            np.testing.assert_allclose(loaded(queries, "query"), fitted(queries, "query"), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(loaded(documents, "data"), fitted(documents, "data"), rtol=1e-5, atol=1e-6)
            self.assertIsNone(loaded.model.outputlayer)
            self.assertIsNone(loaded.selectedepoch)
            self.assertIsNone(loaded.selectedloss)
            self.assertIsNone(loaded.selectionmetric)

            with self.assertRaisesRegex(ValueError, "LEMUR training readout is not available in a loaded inference artifact"):
                loaded.model(torch.ones((1, 6)))

            config = dict(loaded.config)
            config["modeltype"] = "invalid"
            with open(os.path.join(output, "config.json"), "w", encoding="utf-8") as target:
                json.dump(config, target)

            with self.assertRaisesRegex(ValueError, "modeltype must be elm or mlp"):
                Lemur(output)

    def testLemurEpochChoice(self):
        """
        Test LEMUR requires an explicit MLP or ELM epoch choice
        """

        random = np.random.default_rng(42)
        documents = [random.normal(size=(5, 6)).astype(np.float32) for _ in range(8)]

        with self.assertRaisesRegex(ValueError, r"epochs must be set explicitly.*epochs=100.*epochs=0"):
            LemurTrainer().fit(documents)

    def testLemurDefaultEquivalence(self):
        """
        Test implicit and explicit fit defaults are numerically equivalent
        """

        random = np.random.default_rng(42)
        documents = [random.normal(size=(5, 6)).astype(np.float32) for _ in range(8)]
        queries = [random.normal(size=(3, 6)).astype(np.float32) for _ in range(2)]
        settings = {
            "epochs": 4,
            "lr": 0.01,
            "batchsize": 8,
            "hiddendim": 12,
            "finalhiddendim": 10,
            "trainsubsetsize": 8,
            "learnsubsetsize": 40,
            "olssamplesize": 24,
            "seed": 42,
        }
        implicit = LemurTrainer().fit(documents, **settings)
        explicit = LemurTrainer().fit(documents, validationsplit=0.0, **settings)

        # Verify portable numerical equivalence instead of cross-platform bit identity.
        np.testing.assert_allclose(implicit(queries, "query"), explicit(queries, "query"), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(implicit(documents, "data"), explicit(documents, "data"), rtol=1e-5, atol=1e-6)

    def testLemurValidationSelection(self):
        """
        Test validation loss selects and records the retained MLP epoch
        """

        random = np.random.default_rng(1)
        documents = [random.normal(size=(5, 6)).astype(np.float32) for _ in range(8)]

        with tempfile.TemporaryDirectory() as output:
            lemur = LemurTrainer().fit(
                documents,
                output=output,
                epochs=20,
                lr=0.03,
                batchsize=8,
                hiddendim=12,
                finalhiddendim=10,
                trainsubsetsize=8,
                learnsubsetsize=40,
                olssamplesize=24,
                validationsplit=0.25,
                seed=7,
            )
            reloaded = Lemur(output)

            self.assertEqual(lemur.selectionmetric, "validationloss")
            self.assertGreaterEqual(lemur.selectedepoch, 1)
            self.assertLess(lemur.selectedepoch, 20)
            self.assertTrue(np.isfinite(lemur.selectedloss))
            self.assertEqual(reloaded.selectedepoch, lemur.selectedepoch)
            self.assertEqual(reloaded.selectedloss, lemur.selectedloss)
            self.assertEqual(reloaded.selectionmetric, lemur.selectionmetric)

    def testLemurProgress(self):
        """
        Test LEMUR training progress reports validation loss and disables non-interactive output
        """

        class Progress:
            """
            Records tqdm options and postfix updates.
            """

            def __init__(self, values, **options):
                self.values = list(values)
                self.options = options
                self.postfixes = []

            def __iter__(self):
                return iter(self.values)

            def set_postfix(self, values):
                """
                Records a progress postfix update.
                """

                self.postfixes.append(values)

        class Stream:
            """
            Provides non-interactive stderr behavior.
            """

            @staticmethod
            def isatty():
                """
                Returns whether the stream is interactive.
                """

                return False

        progress = []

        def create(values, **options):
            current = Progress(values, **options)
            progress.append(current)
            return current

        random = np.random.default_rng(42)
        documents = [random.normal(size=(5, 6)).astype(np.float32) for _ in range(8)]
        with (
            patch("txtai.pipeline.train.lemur.sys.stderr", new=Stream()),
            patch("txtai.pipeline.train.lemur.tqdm", side_effect=create),
        ):
            lemur = LemurTrainer().fit(
                documents,
                epochs=2,
                lr=0.01,
                batchsize=8,
                hiddendim=12,
                finalhiddendim=10,
                trainsubsetsize=8,
                learnsubsetsize=40,
                olssamplesize=24,
                validationsplit=0.25,
                seed=42,
            )

        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].values, [0, 1])
        self.assertEqual(progress[0].options, {"desc": "LEMUR training", "unit": "epoch", "disable": True})
        self.assertEqual(len(progress[0].postfixes), 2)
        self.assertTrue(all(set(postfix) == {"validation loss"} for postfix in progress[0].postfixes))
        self.assertEqual(lemur.selectionmetric, "validationloss")

    def testLemurRanking(self):
        """
        Test LEMUR ranking quality on pinned synthetic data
        """

        np.random.seed(42)
        torch.manual_seed(42)

        documents = []
        for _ in range(64):
            vectors = np.random.normal(size=(np.random.randint(4, 13), 32)).astype(np.float32)
            documents.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))

        targets = np.random.choice(64, size=8, replace=False)
        queries = []
        for target in targets:
            vectors = documents[target] + np.random.normal(0.0, 0.1, size=documents[target].shape).astype(np.float32)
            queries.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))

        exact = np.asarray([[np.einsum("qd,nd->qn", query, document).max(axis=1).sum() for document in documents] for query in queries])
        exact = np.argsort(-exact, axis=1)

        lemur = LemurTrainer().fit(
            documents,
            epochs=0,
            finalhiddendim=256,
            trainsubsetsize=64,
            learnsubsetsize=sum(len(document) for document in documents),
            olssamplesize=sum(len(document) for document in documents),
            seed=42,
        )
        approximate = lemur(queries, "query") @ lemur(documents, "data").T
        approximate = np.argsort(-approximate, axis=1)

        overlap = np.mean([len(set(exact[x, :10]) & set(approximate[x, :10])) / 10 for x in range(8)])
        top1 = np.sum(exact[:, 0] == approximate[:, 0])

        self.assertGreaterEqual(overlap, 0.6)
        self.assertGreaterEqual(top1, 6)

    def testLemurSettingsValidation(self):
        """
        Test LEMUR rejects non-positive settings
        """

        random = np.random.default_rng(42)
        documents = [random.normal(size=(5, 6)).astype(np.float32) for _ in range(8)]
        settings = [
            "olssamplesize",
            "queryscale",
            "layers",
            "batchsize",
            "trainsubsetsize",
            "learnsubsetsize",
        ]

        for setting in settings:
            for value in (0, -1):
                with self.subTest(setting=setting, value=value):
                    with self.assertRaisesRegex(ValueError, f"{setting} must be greater than 0"):
                        LemurTrainer().fit(documents, epochs=0, **{setting: value})

    def testLemurFitValidation(self):
        """
        Test LEMUR rejects invalid fit inputs
        """

        documents = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[0.5, 0.5]], dtype=np.float32),
        ]
        tests = [
            ("epochs", documents, {"epochs": -1}, "epochs must be greater than or equal to 0"),
            ("final hidden dimension", documents, {"epochs": 0, "finalhiddendim": 0}, "finalhiddendim must be greater than 0"),
            (
                "validation split",
                documents,
                {"epochs": 0, "validationsplit": 1.0},
                "validationsplit must be greater than or equal to 0 and less than 1",
            ),
            ("empty data", [], {"epochs": 0}, "data must contain at least one document"),
            (
                "data dimension",
                [np.ones((1, 2), dtype=np.float32), np.ones((1, 3), dtype=np.float32)],
                {"epochs": 0},
                "all token vectors must have the same dimension",
            ),
            ("empty learn", documents, {"epochs": 0, "learn": []}, "learn must contain at least one document"),
            (
                "learn dimension",
                documents,
                {"epochs": 0, "learn": [np.ones((1, 3), dtype=np.float32)]},
                "all learn token vectors must match the data dimension",
            ),
            (
                "validation training subset",
                documents,
                {"epochs": 0, "learn": [np.ones((1, 2), dtype=np.float32)], "validationsplit": 0.5},
                "validationsplit must leave at least one learn token for training",
            ),
            (
                "zero variance",
                [np.ones((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32)],
                {"epochs": 0},
                "LEMUR targets have zero variance",
            ),
        ]

        for name, data, settings, message in tests:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    LemurTrainer().fit(data, **settings)

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

    def testPEFT(self):
        """
        Test training a model with causal language modeling and PEFT
        """

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
        self.assertTrue("onion" in questions(["What ingredient?"], ["Peel 1 onion"])[0])

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
        model, _ = trainer("hf-internal-testing/tiny-random-electra", self.data, task="token-detection", output_dir=output)

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

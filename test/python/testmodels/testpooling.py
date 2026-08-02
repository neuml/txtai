"""
Pooling module tests
"""

# pylint: disable=too-many-public-methods

import json
import os
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file
import torch

from txtai.models import Models, ClsPooling, LastPooling, LatePooling, Lemur, MeanPooling, PoolingFactory
from txtai.models.pooling.lemur import Activation
from txtai.pipeline import LemurTrainer


class TestPooling(unittest.TestCase):
    """
    Pooling tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize device
        """

        # Device id
        cls.device = Models.deviceid(True)

    def testCLS(self):
        """
        Test CLS pooling
        """

        # Test CLS pooling
        pooling = PoolingFactory.create({"path": "flax-sentence-embeddings/multi-qa_v1-MiniLM-L6-cls_dot", "device": self.device})
        self.assertEqual(type(pooling), ClsPooling)

        pooling = PoolingFactory.create({"method": "clspooling", "path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device})
        self.assertEqual(type(pooling), ClsPooling)

        # Test CLS pooling encoding
        self.assertEqual(pooling.encode(["test"])[0].shape, (768,))

    def testLast(self):
        """
        Test last pooling
        """

        # Test last pooling
        pooling = PoolingFactory.create({"path": "neuml/bert-tiny-sts-last-pooling", "device": self.device})
        self.assertEqual(type(pooling), LastPooling)

        pooling = PoolingFactory.create({"method": "lastpooling", "path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device})
        self.assertEqual(type(pooling), LastPooling)

        # Test last pooling encoding
        self.assertEqual(pooling.encode(["test"])[0].shape, (768,))

    def testLength(self):
        """
        Test pooling with max_seq_length
        """

        # Test reading max_seq_length parmaeter
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device, "maxlength": True})
        self.assertEqual(pooling.maxlength, 75)

        # Test specified maxlength
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device, "maxlength": 256})
        self.assertEqual(pooling.maxlength, 256)

        # Test max_seq_length is ignored when parameter is omitted
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device})
        self.assertEqual(pooling.maxlength, 512)

        # Test maxlength when max_seq_length not present
        pooling = PoolingFactory.create({"path": "hf-internal-testing/tiny-random-gpt2", "device": self.device, "maxlength": True})
        self.assertEqual(pooling.maxlength, 1024)

    def testMean(self):
        """
        Test mean pooling
        """

        # Test mean pooling
        pooling = PoolingFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device})
        self.assertEqual(type(pooling), MeanPooling)

        pooling = PoolingFactory.create(
            {"method": "meanpooling", "path": "flax-sentence-embeddings/multi-qa_v1-MiniLM-L6-cls_dot", "device": self.device}
        )
        self.assertEqual(type(pooling), MeanPooling)

    def testMuvera(self):
        """
        Test late pooling with MUVERA fixed dimensional encoding
        """

        # Test MUVERA encoding
        for model in ["neuml/colbert-bert-tiny", "neuml/pylate-bert-tiny"]:
            # Test defaults
            pooling = PoolingFactory.create({"path": model, "device": self.device})
            self.assertEqual(pooling.encode(["test"], category="query").shape, (1, 10240))

            # Test custom settings
            pooling = PoolingFactory.create(
                {"path": model, "device": self.device, "modelargs": {"muvera": {"repetitions": 5, "hashes": 2, "projection": 8}}}
            )
            self.assertEqual(pooling.encode(["test"], category="data").shape, (1, 160))

    def testLateCenterDefaults(self):
        """
        Test late pooling token centering defaults
        """

        empty = torch.nn.Sequential()
        single = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        multiple = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)))

        self.assertIsNone(LatePooling.centersettings(None, empty, False))
        self.assertIsNone(LatePooling.centersettings(None, single, False))
        self.assertEqual(LatePooling.centersettings(None, multiple, False), {"scope": "batch"})
        self.assertIsNone(LatePooling.centersettings(False, multiple, True))
        self.assertEqual(LatePooling.centersettings(True, empty, True), {"scope": "batch"})
        self.assertEqual(LatePooling.centersettings({"scope": "batch"}, empty, True), {"scope": "batch"})
        self.assertEqual(LatePooling.centersettings({"scope": "document"}, empty, True), {"scope": "document"})

    def testLateCenterSettings(self):
        """
        Test late pooling token centering settings
        """

        linear = torch.nn.Sequential()
        mean = np.array([0.25, -0.25], dtype=np.float32)
        settings = LatePooling.centersettings({"scope": "collection", "mean": mean.tolist()}, linear, True)
        np.testing.assert_array_equal(settings["mean"], mean)

        with tempfile.TemporaryDirectory() as output:
            path = os.path.join(output, "mean.safetensors")
            save_file({"center.mean": mean}, path)
            settings = LatePooling.centersettings({"scope": "collection", "path": path}, linear, True)
            np.testing.assert_array_equal(settings["mean"], mean)

        tests = [
            (None, "center must be a boolean or dictionary"),
            ({"scope": "invalid"}, "center scope must be one of"),
            ({"scope": "collection"}, "requires exactly one"),
            ({"scope": "collection", "mean": mean, "path": "mean.safetensors"}, "requires exactly one"),
            ({"scope": "document", "mean": mean}, "only valid with collection scope"),
            ({"scope": "document", "invalid": True}, "unknown center setting"),
            ({"scope": "collection", "mean": [[0.0, 1.0]]}, "finite one-dimensional array"),
        ]

        for center, message in tests:
            with self.subTest(center=center):
                with self.assertRaisesRegex(ValueError, message):
                    LatePooling.centersettings(center, linear, True)

    def testLateCenterScopes(self):
        """
        Test document, batch and collection token centering
        """

        data = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        outputs = {}
        for scope in ("document", "batch", "collection"):
            pooling = LatePooling.__new__(LatePooling)
            object.__setattr__(pooling, "center", {"scope": scope, "mean": np.array([0.25, 0.25])} if scope == "collection" else {"scope": scope})
            outputs[scope] = pooling.centerdata(data)

            np.testing.assert_array_equal(outputs[scope][:, 2], np.zeros((2, 2), dtype=np.float32))
            norms = np.linalg.norm(outputs[scope][:, :2], axis=2)
            self.assertTrue(np.all((norms == 0.0) | np.isclose(norms, 1.0)))

        self.assertFalse(np.array_equal(outputs["document"], outputs["batch"]))
        self.assertFalse(np.array_equal(outputs["batch"], outputs["collection"]))

        # Document centering is batch-independent and equals batch centering for one item
        pooling = LatePooling.__new__(LatePooling)
        object.__setattr__(pooling, "center", {"scope": "document"})
        separate = np.vstack([pooling.centerdata(data[x : x + 1]) for x in range(len(data))])
        np.testing.assert_array_equal(outputs["document"], separate)

        object.__setattr__(pooling, "center", {"scope": "batch"})
        object.__setattr__(pooling, "batches", [1, 1])
        for x in range(len(data)):
            np.testing.assert_array_equal(pooling.centerdata(data[x : x + 1]), outputs["document"][x : x + 1])
        np.testing.assert_array_equal(pooling.centerdata(data), outputs["document"])

        object.__setattr__(pooling, "center", {"scope": "document"})
        object.__setattr__(pooling, "encoder", None)
        object.__setattr__(pooling, "lengths", [2, 2])
        query = pooling.postencode([value.copy() for value in data[:, :2]], "query")
        documents = pooling.postencode([value.copy() for value in data[:, :2]], "data")
        np.testing.assert_array_equal(query, documents)

        object.__setattr__(pooling, "center", {"scope": "collection", "mean": np.zeros(3)})
        with self.assertRaisesRegex(ValueError, "dimension must match"):
            pooling.centerdata(data)

    def testLateCenterDisabled(self):
        """
        Test omitted and explicitly disabled centering are byte-identical
        """

        base = PoolingFactory.create({"path": "neuml/colbert-bert-tiny", "device": self.device, "modelargs": {"muvera": None}})
        disabled = PoolingFactory.create({"path": "neuml/colbert-bert-tiny", "device": self.device, "modelargs": {"muvera": None, "center": False}})

        self.assertIsNone(base.center)
        self.assertIsNone(disabled.center)
        np.testing.assert_array_equal(base.encode(["test"], category="query"), disabled.encode(["test"], category="query"))

        centered = PoolingFactory.create({"path": "neuml/colbert-bert-tiny", "device": self.device, "modelargs": {"muvera": None, "center": True}})
        texts = ["Short text.", "A considerably longer text exercises padding behavior."]
        self.assertEqual(centered.center, {"scope": "batch"})
        together = centered.encode(texts, batch=2, category="data")
        repeated = centered.encode(texts, batch=2, category="data")
        np.testing.assert_array_equal(together, repeated)
        self.assertEqual(centered.batches, [2])
        self.assertTrue(np.any(np.all(together == 0.0, axis=2)))

        centered.center = {"scope": "document"}
        separate = centered.encode(texts, batch=1, category="data")
        document = centered.encode(texts, batch=2, category="data")
        np.testing.assert_allclose(document, separate, rtol=1e-4, atol=1e-5)

    def testLemur(self):
        """
        Test late pooling with LEMUR fixed dimensional encoding
        """

        corpus = [
            "Machine learning models retrieve relevant passages.",
            "Late interaction compares token embeddings.",
            "Dense indexes search fixed dimensional vectors.",
            "A query encoder produces contextual token vectors.",
            "Document encoders represent passages for retrieval.",
            "Maximum similarity aggregates token matches.",
            "LEMUR learns a corpus specific reduction.",
            "MUVERA uses randomized fixed dimensional encodings.",
            "The trainer stores reusable pooling artifacts.",
            "New documents can be encoded after training.",
            "Short text.",
            "A considerably longer synthetic document exercises padding behavior.",
        ] * 2

        for model in ["neuml/colbert-bert-tiny", "neuml/pylate-bert-tiny"]:
            with tempfile.TemporaryDirectory() as output:
                LemurTrainer()(
                    model,
                    corpus,
                    output,
                    gpu=False,
                    epochs=0,
                    finalhiddendim=128,
                    trainsubsetsize=24,
                    learnsubsetsize=256,
                    olssamplesize=128,
                    seed=42,
                )

                pooling = PoolingFactory.create({"path": model, "device": self.device, "modelargs": {"lemur": {"path": output}, "center": False}})
                texts = ["Short text.", "A considerably longer synthetic document exercises padding behavior."]
                queries = pooling.encode(texts, category="query")
                documents = pooling.encode(texts, category="data")

                self.assertEqual(queries.shape, (2, 128))
                self.assertEqual(documents.shape, (2, 128))
                self.assertTrue(np.isfinite(queries).all())
                self.assertTrue(np.isfinite(documents).all())

                # LEMUR must use true token counts, independent of batch padding
                singles = np.vstack([pooling.encode([text], category="data") for text in texts])
                np.testing.assert_allclose(documents, singles, rtol=1e-4, atol=1e-5)

            # MUVERA remains the default when LEMUR is absent
            pooling = PoolingFactory.create({"path": model, "device": self.device})
            self.assertEqual(pooling.encode(["test"], category="query").shape, (1, 10240))

    def testLemurCenter(self):
        """
        Test late pooling with centered LEMUR fixed dimensional encoding
        """

        corpus = [
            "Machine learning models retrieve relevant passages.",
            "Late interaction compares token embeddings.",
            "Dense indexes search fixed dimensional vectors.",
            "A query encoder produces contextual token vectors.",
            "Document encoders represent passages for retrieval.",
            "Maximum similarity aggregates token matches.",
            "LEMUR learns a corpus specific reduction.",
            "MUVERA uses randomized fixed dimensional encodings.",
            "The trainer stores reusable pooling artifacts.",
            "New documents can be encoded after training.",
            "Short text.",
            "A considerably longer synthetic document exercises padding behavior.",
        ] * 2
        texts = ["Short text.", "A considerably longer synthetic document exercises padding behavior."]

        for model in ["neuml/colbert-bert-tiny", "neuml/pylate-bert-tiny"]:
            with tempfile.TemporaryDirectory() as output:
                LemurTrainer()(
                    model,
                    corpus,
                    output,
                    gpu=False,
                    epochs=0,
                    finalhiddendim=128,
                    trainsubsetsize=24,
                    learnsubsetsize=256,
                    olssamplesize=128,
                    seed=42,
                )

                pooling = PoolingFactory.create({"path": model, "device": self.device, "modelargs": {"lemur": {"path": output}, "center": True}})
                self.assertEqual(pooling.center, {"scope": "batch"})

                queries = pooling.encode(texts, batch=2, category="query")
                documents = pooling.encode(texts, batch=2, category="data")
                self.assertEqual(queries.shape, (2, 128))
                self.assertEqual(documents.shape, (2, 128))
                self.assertTrue(np.isfinite(queries).all())
                self.assertTrue(np.isfinite(documents).all())

                np.testing.assert_array_equal(pooling.encode(texts, batch=2, category="query"), queries)
                np.testing.assert_array_equal(pooling.encode(texts, batch=2, category="data"), documents)

    def testLemurRoundTrip(self):
        """
        Test an ordinary LEMUR artifact save/load round-trip
        """

        random = np.random.default_rng(42)
        documents = [random.normal(size=(5, 6)).astype(np.float32) for _ in range(8)]
        queries = [random.normal(size=(3, 6)).astype(np.float32) for _ in range(2)]

        with tempfile.TemporaryDirectory() as output:
            fitted = Lemur().fit(
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

    def testLemurDocuments(self):
        """
        Test LEMUR document input conversions and validation
        """

        lemur = Lemur()
        matrix = np.ones((2, 3), dtype=np.float32)
        batches = np.ones((2, 2, 3), dtype=np.float32)

        documents = lemur.documents(matrix)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].shape, (2, 3))

        documents = lemur.documents(batches)
        self.assertEqual(len(documents), 2)
        self.assertTrue(all(document.shape == (2, 3) for document in documents))

        tensor = torch.ones((2, 3), dtype=torch.float64)
        document = lemur.documents([tensor])[0]
        self.assertEqual(document.dtype, torch.float32)
        self.assertEqual(document.shape, (2, 3))

        document = lemur.documents([[[1.0, 2.0], [3.0, 4.0]]])[0]
        self.assertEqual(document.shape, (2, 2))

        for invalid in ([np.ones(3, dtype=np.float32)], [np.empty((0, 3), dtype=np.float32)]):
            with self.subTest(shape=invalid[0].shape):
                with self.assertRaisesRegex(ValueError, "each document must be a non-empty 2D token-vector array"):
                    lemur.documents(invalid)

    def testLemurEpochChoice(self):
        """
        Test LEMUR requires an explicit MLP or ELM epoch choice
        """

        random = np.random.default_rng(42)
        documents = [random.normal(size=(5, 6)).astype(np.float32) for _ in range(8)]

        with self.assertRaisesRegex(ValueError, r"epochs must be set explicitly.*epochs=100.*epochs=0"):
            Lemur().fit(documents)

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
        implicit = Lemur().fit(documents, **settings)
        explicit = Lemur().fit(documents, validationsplit=0.0, **settings)

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
            lemur = Lemur().fit(
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

        lemur = Lemur()
        lemur.fit(
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
                        Lemur().fit(documents, epochs=0, **{setting: value})

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
                    Lemur().fit(data, **settings)

    def testLemurStateValidation(self):
        """
        Test LEMUR rejects invalid categories and unfitted encoding
        """

        lemur = Lemur()
        documents = [np.ones((1, 2), dtype=np.float32)]

        with self.assertRaisesRegex(ValueError, "category must be query or data"):
            lemur(documents, "invalid")

        with self.assertRaisesRegex(ValueError, "LEMUR must be fitted or loaded before encoding"):
            lemur(documents, "query")

    def testLemurActivations(self):
        """
        Test LEMUR activations resolve to Torch modules and functions
        """

        data = torch.tensor([-1.0, 0.0, 0.5, 2.0])
        expected = {
            "relu": (torch.nn.ReLU, torch.relu),
            "gelu": (torch.nn.GELU, torch.nn.functional.gelu),
            "silu": (torch.nn.SiLU, torch.nn.functional.silu),
            "mish": (torch.nn.Mish, torch.nn.functional.mish),
        }

        for name, (module, function) in expected.items():
            with self.subTest(activation=name):
                self.assertIsInstance(Activation.module(name), module)
                self.assertTrue(torch.equal(Activation.function(name)(data), function(data)))

        for method in (Activation.module, Activation.function):
            with self.assertRaisesRegex(ValueError, "activation must be one of: relu, gelu, silu, mish"):
                method("invalid")

    def testPrompts(self):
        """
        Test instruction prompts
        """

        # Load model with prompts
        pooling = PoolingFactory.create({"path": "neuml/bert-tiny-prompts", "device": self.device, "loadprompts": True})

        # Test prompts are prepended
        self.assertEqual(pooling.preencode(["abc"], "query")[0], "query: abc")
        self.assertEqual(pooling.preencode(["text"], "data")[0], "document: text")

        # Load model with prompts disabled (default)
        pooling = PoolingFactory.create({"path": "neuml/bert-tiny-prompts", "device": self.device})

        # Test that prompts are not prepended
        self.assertEqual(pooling.preencode(["abc"], "query")[0], "abc")
        self.assertEqual(pooling.preencode(["text"], "data")[0], "text")

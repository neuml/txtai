"""
Pooling module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

import numpy as np
import torch

from safetensors.numpy import save_file

from txtai.models import Models, ClsPooling, LastPooling, LatePooling, Lemur, MaxPooling, MeanPooling, Muvera, PoolingFactory
from txtai.models.pooling.lemur import Activation
from txtai.pipeline import LemurTrainer


# pylint: disable=R0904
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

    def testLateCenterDefaults(self):
        """
        Test late pooling token centering defaults
        """

        empty = torch.nn.Sequential()
        single = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        multiple = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)))

        # Create pooling instance
        pool = PoolingFactory.create({"path": "neuml/colbert-bert-tiny", "device": self.device, "modelargs": {"muvera": None}})

        self.assertIsNone(pool.centersettings(None, empty, False))
        self.assertIsNone(pool.centersettings(None, single, False))
        self.assertEqual(pool.centersettings(None, multiple, False), {"scope": "batch"})
        self.assertIsNone(pool.centersettings(False, multiple, True))
        self.assertEqual(pool.centersettings(True, empty, True), {"scope": "batch"})
        self.assertEqual(pool.centersettings({"scope": "batch"}, empty, True), {"scope": "batch"})
        self.assertEqual(pool.centersettings({"scope": "document"}, empty, True), {"scope": "document"})

    def testLateEncoderSettings(self):
        """
        Test late pooling fixed encoder settings
        """

        pool = PoolingFactory.create({"path": "neuml/colbert-bert-tiny", "device": self.device, "modelargs": {"muvera": None}})
        lemur = {"path": "artifact"}
        muvera = {"repetitions": 5}

        self.assertIsNone(pool.encodersettings(None, "lemur"))
        self.assertIsNone(pool.encodersettings(False, "lemur"))
        self.assertEqual(pool.encodersettings("artifact", "lemur"), {"path": "artifact"})
        self.assertIs(pool.encodersettings(lemur, "lemur"), lemur)

        self.assertIsNone(pool.encodersettings(None, "muvera"))
        self.assertIsNone(pool.encodersettings(False, "muvera"))
        self.assertEqual(pool.encodersettings(True, "muvera"), {})
        self.assertIs(pool.encodersettings(muvera, "muvera"), muvera)

        for value in (True, 1):
            with self.subTest(lemur=value):
                with self.assertRaisesRegex(ValueError, "^lemur expects a path string or a dict of settings$"):
                    pool.encodersettings(value, "lemur")

        for value in ("settings", 1):
            with self.subTest(muvera=value):
                with self.assertRaisesRegex(ValueError, "^muvera expects a boolean or a dict of settings$"):
                    pool.encodersettings(value, "muvera")

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

    def testLateCenterSettings(self):
        """
        Test late pooling token centering settings
        """

        # Create pooling instance
        pool = PoolingFactory.create({"path": "neuml/colbert-bert-tiny", "device": self.device, "modelargs": {"muvera": None}})

        linear = torch.nn.Sequential()
        mean = np.array([0.25, -0.25], dtype=np.float32)
        settings = pool.centersettings({"scope": "collection", "mean": mean.tolist()}, linear, True)
        np.testing.assert_array_equal(settings["mean"], mean)

        with tempfile.TemporaryDirectory() as output:
            path = os.path.join(output, "mean.safetensors")
            save_file({"center.mean": mean}, path)
            settings = pool.centersettings({"scope": "collection", "path": path}, linear, True)
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
                    pool.centersettings(center, linear, True)

    def testLateSettings(self):
        """
        Test late pooling defaults when the settings file is missing
        """

        pool = PoolingFactory.create({"path": "neuml/colbert-bert-tiny", "device": self.device, "modelargs": {"muvera": None}})

        # A missing config_sentence_transformers.json (PyLate) or artifact.metadata (Stanford) reads as default settings
        with patch.object(LatePooling, "load", return_value=None):
            self.assertEqual(pool.settings("neuml/colbert-bert-tiny", [{"path": "1_Dense"}]), [None] * 4)
            self.assertEqual(pool.settings("neuml/colbert-bert-tiny", None), [None] * 4)

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

                pathpooling = PoolingFactory.create({"path": model, "device": self.device, "modelargs": {"lemur": output, "center": False}})
                self.assertIsInstance(pathpooling.encoder, Lemur)
                np.testing.assert_allclose(pathpooling.encode(texts, category="data"), documents, rtol=1e-4, atol=1e-5)

                # LEMUR must use true token counts, independent of batch padding
                singles = np.vstack([pooling.encode([text], category="data") for text in texts])
                np.testing.assert_allclose(documents, singles, rtol=1e-4, atol=1e-5)

            # MUVERA remains the default when LEMUR is absent
            pooling = PoolingFactory.create({"path": model, "device": self.device})
            self.assertEqual(pooling.encode(["test"], category="query").shape, (1, 10240))

            pooling = PoolingFactory.create({"path": model, "device": self.device, "modelargs": {"muvera": False}})
            self.assertIsNone(pooling.encoder)

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

    def testLemurCollectionCenter(self):
        """
        Test LEMUR artifact collection centering is batch-independent
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
        ]
        texts = corpus[-2:]

        def loadlinear(*_):
            layers = [torch.nn.Linear(128, 128, bias=False) for _ in range(2)]
            with torch.no_grad():
                for layer in layers:
                    layer.weight.copy_(torch.eye(128))
            return torch.nn.Sequential(*layers)

        settings = {
            "gpu": False,
            "epochs": 0,
            "finalhiddendim": 32,
            "trainsubsetsize": 12,
            "learnsubsetsize": 128,
            "olssamplesize": 64,
            "seed": 42,
        }

        with (
            tempfile.TemporaryDirectory() as output,
            tempfile.TemporaryDirectory() as legacy,
            patch.object(LatePooling, "loadlinear", autospec=True, side_effect=loadlinear),
        ):
            raw = PoolingFactory.create(
                {
                    "path": "neuml/colbert-bert-tiny",
                    "device": Models.deviceid(False),
                    "modelargs": {"muvera": None, "lemur": None, "center": False},
                }
            )
            documents = [raw.encode([text], batch=1, category="data")[0] for text in corpus]
            center = np.concatenate(documents).mean(axis=0)

            LemurTrainer()(
                "neuml/colbert-bert-tiny",
                corpus,
                output,
                **settings,
            )
            LemurTrainer()("neuml/colbert-bert-tiny", corpus, legacy, vectors={"center": False}, **settings)

            pooling = PoolingFactory.create(
                {
                    "path": "neuml/colbert-bert-tiny",
                    "device": Models.deviceid(False),
                    "modelargs": {"lemur": {"path": output}},
                }
            )
            self.assertEqual(pooling.center["scope"], "collection")
            np.testing.assert_array_equal(pooling.center["mean"], center)
            np.testing.assert_array_equal(pooling.encoder.center.cpu().numpy(), center)

            separate = pooling.encode(texts, batch=1, category="data")
            together = pooling.encode(texts, batch=32, category="data")
            np.testing.assert_allclose(together, separate, rtol=1e-4, atol=1e-5)

            separate = pooling.encode(texts, batch=1, category="query")
            together = pooling.encode(texts, batch=32, category="query")
            np.testing.assert_allclose(together, separate, rtol=1e-4, atol=1e-5)

            disabled = PoolingFactory.create(
                {
                    "path": "neuml/colbert-bert-tiny",
                    "device": Models.deviceid(False),
                    "modelargs": {"lemur": {"path": output}, "center": False},
                }
            )
            self.assertIsNone(disabled.center)
            self.assertIsNotNone(disabled.encoder.center)

            compatible = PoolingFactory.create(
                {
                    "path": "neuml/colbert-bert-tiny",
                    "device": Models.deviceid(False),
                    "modelargs": {"lemur": {"path": legacy}},
                }
            )
            self.assertEqual(compatible.center, {"scope": "batch"})
            self.assertIsNone(compatible.encoder.center)

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

    def testLemurPathValidation(self):
        """
        Test LEMUR rejects a path that is not a LEMUR artifact
        """

        # An encoder repo also has config.json and model.safetensors
        for model in ["neuml/colbert-bert-tiny", "neuml/pylate-bert-tiny"]:
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "not a LEMUR artifact"):
                    PoolingFactory.create({"path": model, "device": self.device, "modelargs": {"lemur": {"path": model}}})

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

    def testMax(self):
        """
        Test max pooling
        """

        # Test max pooling
        pooling = PoolingFactory.create({"method": "maxpooling", "path": "sentence-transformers/nli-mpnet-base-v2", "device": self.device})
        self.assertEqual(type(pooling), MaxPooling)

        # Test max pooling encoding
        self.assertEqual(pooling.encode(["test"])[0].shape, (768,))

        # Padding tokens must be excluded from the max, otherwise a short text changes when batched with a longer one
        texts = ["Short text.", "A considerably longer text exercises padding behavior."]
        self.assertTrue(np.allclose(pooling.encode([texts[0]])[0], pooling.encode(texts, batch=2)[0], atol=1e-4))

    def testMaxMethod(self):
        """
        Test max pooling is derived from the sentence transformers pooling config
        """

        config = {"pooling_mode_max_tokens": True, "pooling_mode_mean_tokens": False}
        with patch.object(PoolingFactory, "load", return_value=config):
            self.assertEqual(PoolingFactory.method("sentence-transformers/nli-mpnet-base-v2"), "maxpooling")

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

    def testMethodPartialConfig(self):
        """
        Test pooling config files that omit pooling_mode_mean_tokens
        """

        # A missing mean flag reads as disabled, same as an explicit false, instead of raising KeyError
        tests = [
            ({"pooling_mode_cls_token": True}, "clspooling"),
            ({"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False}, "clspooling"),
            ({"pooling_mode_lasttoken": True}, "lastpooling"),
            ({"pooling_mode_lasttoken": True, "pooling_mode_mean_tokens": False}, "lastpooling"),
            ({"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": True}, "meanpooling"),
            ({}, "meanpooling"),
        ]

        for config, expected in tests:
            with patch.object(PoolingFactory, "load", return_value=config):
                self.assertEqual(PoolingFactory.method("sentence-transformers/nli-mpnet-base-v2"), expected)

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

    def testMuveraPadding(self):
        """
        Test MUVERA vectors don't change with batch padding
        """

        pooling = PoolingFactory.create({"path": "neuml/colbert-bert-tiny", "device": self.device})
        texts = ["Short text.", "A considerably longer text exercises padding behavior."]

        # The shorter text must encode the same alone as it does batched with a longer text
        for category in ["query", "data"]:
            alone = pooling.encode([texts[0]], category=category)
            batched = pooling.encode(texts, batch=2, category=category)
            self.assertTrue(np.allclose(alone[0], batched[0], atol=1e-4))

    def testMuveraTorchMatchesNumPy(self):
        """
        Test that the Torch MUVERA implementation produces the same encodings as the NumPy one
        """

        # Deterministic multi-vector input: three documents of varying token counts
        rng = np.random.default_rng(1234)
        data = [rng.standard_normal((n, 32)).astype(np.float32) for n in (5, 11, 3)]

        muvera = Muvera(repetitions=4, hashes=3, projection=8, seed=42)

        outputs = muvera(data, "data")

        # Output width must be repetitions * 2^hashes * projection
        self.assertEqual(outputs.shape, (3, 4 * (2**3) * 8))

        # Encoding must be deterministic for a fixed seed
        self.assertTrue(np.allclose(outputs, muvera(data, "data"), atol=1e-5))

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

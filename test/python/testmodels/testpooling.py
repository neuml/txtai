"""
Pooling module tests
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from txtai.models import Models, ClsPooling, LastPooling, Lemur, MeanPooling, PoolingFactory
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
                    final_hidden_dim=128,
                    train_subset_size=24,
                    learn_subset_size=256,
                    ols_sample_size=128,
                    seed=42,
                )

                pooling = PoolingFactory.create({"path": model, "device": self.device, "modelargs": {"lemur": {"path": output}}})
                texts = ["Short text.", "A considerably longer synthetic document exercises padding behavior."]
                queries = pooling.encode(texts, category="query")
                documents = pooling.encode(texts, category="data")

                self.assertEqual(queries.shape, (2, 128))
                self.assertEqual(documents.shape, (2, 128))
                self.assertTrue(np.isfinite(queries).all())
                self.assertTrue(np.isfinite(documents).all())

                # LEMUR must use true token counts, independent of batch padding
                singles = np.vstack([pooling.encode([text], category="data") for text in texts])
                np.testing.assert_allclose(documents, singles, rtol=1e-5, atol=1e-6)

            # MUVERA remains the default when LEMUR is absent
            pooling = PoolingFactory.create({"path": model, "device": self.device})
            self.assertEqual(pooling.encode(["test"], category="query").shape, (1, 10240))

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
                final_hidden_dim=10,
                train_subset_size=8,
                learn_subset_size=40,
                ols_sample_size=24,
                seed=42,
            )
            self.assertEqual(set(os.listdir(output)), {"config.json", "model.safetensors"})
            loaded = Lemur(output)

            # Float32 feature and SVD kernels can vary across Torch/BLAS builds.
            np.testing.assert_allclose(loaded(queries, "query"), fitted(queries, "query"), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(loaded(documents, "data"), fitted(documents, "data"), rtol=1e-5, atol=1e-6)
            self.assertIsNone(loaded.model.output_layer)
            self.assertIsNone(loaded.selected_epoch)
            self.assertIsNone(loaded.selected_loss)
            self.assertIsNone(loaded.selection_metric)

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
            "batch_size": 8,
            "hidden_dim": 12,
            "final_hidden_dim": 10,
            "train_subset_size": 8,
            "learn_subset_size": 40,
            "ols_sample_size": 24,
            "seed": 42,
        }
        implicit = Lemur().fit(documents, **settings)
        explicit = Lemur().fit(documents, validation_split=0.0, **settings)

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
                batch_size=8,
                hidden_dim=12,
                final_hidden_dim=10,
                train_subset_size=8,
                learn_subset_size=40,
                ols_sample_size=24,
                validation_split=0.25,
                seed=7,
            )
            reloaded = Lemur(output)

            self.assertEqual(lemur.selection_metric, "validation_loss")
            self.assertGreaterEqual(lemur.selected_epoch, 1)
            self.assertLess(lemur.selected_epoch, 20)
            self.assertTrue(np.isfinite(lemur.selected_loss))
            self.assertEqual(reloaded.selected_epoch, lemur.selected_epoch)
            self.assertEqual(reloaded.selected_loss, lemur.selected_loss)
            self.assertEqual(reloaded.selection_metric, lemur.selection_metric)

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
            final_hidden_dim=256,
            train_subset_size=64,
            learn_subset_size=sum(len(document) for document in documents),
            ols_sample_size=sum(len(document) for document in documents),
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
            "ols_sample_size",
            "query_scale",
            "num_layers",
            "batch_size",
            "train_subset_size",
            "learn_subset_size",
        ]

        for setting in settings:
            for value in (0, -1):
                with self.subTest(setting=setting, value=value):
                    with self.assertRaisesRegex(ValueError, f"{setting} must be greater than 0"):
                        Lemur().fit(documents, epochs=0, **{setting: value})

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

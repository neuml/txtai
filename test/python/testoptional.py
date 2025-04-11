"""
Optional module tests
"""

import sys
import unittest

# pylint: disable=C0415,W0611,W0621
import timm
import txtai


class TestOptional(unittest.TestCase):
    """
    Optional tests. Simulates optional dependencies not being installed.
    """

    @classmethod
    def setUpClass(cls):
        """
        Simulate optional packages not being installed
        """

        modules = [
            "annoy",
            "bs4",
            "chonkie",
            "croniter",
            "docling.document_converter",
            "duckdb",
            "fastapi",
            "gliner",
            "grandcypher",
            "grand",
            "hnswlib",
            "imagehash",
            "libcloud.storage.providers",
            "litellm",
            "llama_cpp",
            "model2vec",
            "networkx",
            "nltk",
            "onnxmltools",
            "onnxruntime",
            "onnxruntime.quantization",
            "pandas",
            "peft",
            "pgvector",
            "PIL",
            "rich",
            "scipy",
            "sentence_transformers",
            "sklearn.decomposition",
            "smolagents",
            "sounddevice",
            "soundfile",
            "sqlalchemy",
            "sqlite_vec",
            "staticvectors",
            "tika",
            "ttstokenizer",
            "xmltodict",
        ]

        # Get handle to all currently loaded txtai modules
        modules = modules + [key for key in sys.modules if key.startswith("txtai")]
        cls.modules = {module: None for module in modules}

        # Replace loaded modules with stubs. Save modules for later reloading
        for module in cls.modules:
            if module in sys.modules:
                cls.modules[module] = sys.modules[module]

            # Remove txtai modules. Set optional dependencies to None to prevent reloading.
            if "txtai" in module:
                if module in sys.modules:
                    del sys.modules[module]
            else:
                sys.modules[module] = None

    @classmethod
    def tearDownClass(cls):
        """
        Resets modules environment back to initial state.
        """

        # Reset replaced modules in setup
        for key, value in cls.modules.items():
            if value:
                sys.modules[key] = value
            else:
                del sys.modules[key]

    def testAgent(self):
        """
        Test missing agent dependencies
        """

        from txtai.agent import Agent

        with self.assertRaises(ImportError):
            Agent(llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

    def testANN(self):
        """
        Test missing ANN dependencies
        """

        from txtai.ann import ANNFactory

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "annoy"})

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "hnsw"})

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "pgvector"})

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "sqlite"})

    def testApi(self):
        """
        Test missing api dependencies
        """

        with self.assertRaises(ImportError):
            import txtai.api

    def testConsole(self):
        """
        Test missing console dependencies
        """

        from txtai.console import Console

        with self.assertRaises(ImportError):
            Console()

    def testCloud(self):
        """
        Test missing cloud dependencies
        """

        from txtai.cloud import ObjectStorage

        with self.assertRaises(ImportError):
            ObjectStorage(None)

    def testDatabase(self):
        """
        Test missing database dependencies
        """

        from txtai.database import Client, DuckDB, ImageEncoder

        with self.assertRaises(ImportError):
            Client({})

        with self.assertRaises(ImportError):
            DuckDB({})

        with self.assertRaises(ImportError):
            ImageEncoder()

    def testGraph(self):
        """
        Test missing graph dependencies
        """

        from txtai.graph import GraphFactory, Query

        with self.assertRaises(ImportError):
            GraphFactory.create({"backend": "networkx"})

        with self.assertRaises(ImportError):
            GraphFactory.create({"backend": "rdbms"})

        with self.assertRaises(ImportError):
            Query()

    def testModel(self):
        """
        Test missing model dependencies
        """

        from txtai.embeddings import Reducer
        from txtai.models import OnnxModel

        with self.assertRaises(ImportError):
            Reducer()

        with self.assertRaises(ImportError):
            OnnxModel(None)

    def testPipeline(self):
        """
        Test missing pipeline dependencies
        """

        from txtai.pipeline import (
            AudioMixer,
            AudioStream,
            Caption,
            Entity,
            FileToHTML,
            HFOnnx,
            HFTrainer,
            HTMLToMarkdown,
            ImageHash,
            LiteLLM,
            LlamaCpp,
            Microphone,
            MLOnnx,
            Objects,
            Segmentation,
            Tabular,
            TextToAudio,
            TextToSpeech,
            Transcription,
            Translation,
        )

        with self.assertRaises(ImportError):
            AudioMixer()

        with self.assertRaises(ImportError):
            AudioStream()

        with self.assertRaises(ImportError):
            Caption()

        with self.assertRaises(ImportError):
            Entity("neuml/gliner-bert-tiny")

        with self.assertRaises(ImportError):
            FileToHTML(backend="docling")

        with self.assertRaises(ImportError):
            FileToHTML(backend="tika")

        with self.assertRaises(ImportError):
            HFOnnx()("google/bert_uncased_L-2_H-128_A-2", quantize=True)

        with self.assertRaises(ImportError):
            HFTrainer()(None, None, lora=True)

        with self.assertRaises(ImportError):
            HTMLToMarkdown()

        with self.assertRaises(ImportError):
            ImageHash()

        with self.assertRaises(ImportError):
            LiteLLM("huggingface/t5-small")

        with self.assertRaises(ImportError):
            LlamaCpp("TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/tinyllama-1.1b-chat-v0.3.Q2_K.gguf")

        with self.assertRaises(ImportError):
            Microphone()

        with self.assertRaises(ImportError):
            MLOnnx()

        with self.assertRaises(ImportError):
            Objects()

        with self.assertRaises(ImportError):
            Segmentation(sentences=True)

        with self.assertRaises(ImportError):
            Segmentation(chunker="token")

        with self.assertRaises(ImportError):
            Tabular()

        with self.assertRaises(ImportError):
            TextToAudio()

        with self.assertRaises(ImportError):
            TextToSpeech()

        with self.assertRaises(ImportError):
            Transcription()

        with self.assertRaises(ImportError):
            Translation().detect(["test"])

    def testScoring(self):
        """
        Test missing scoring dependencies
        """

        from txtai.scoring import ScoringFactory

        with self.assertRaises(ImportError):
            ScoringFactory.create({"method": "pgtext"})

    def testVectors(self):
        """
        Test missing vector dependencies
        """

        from txtai.vectors import VectorsFactory

        with self.assertRaises(ImportError):
            VectorsFactory.create({"method": "litellm", "path": "huggingface/sentence-transformers/all-MiniLM-L6-v2"}, None)

        with self.assertRaises(ImportError):
            VectorsFactory.create({"method": "llama.cpp", "path": "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q2_K.gguf"}, None)

        with self.assertRaises(ImportError):
            VectorsFactory.create({"method": "model2vec", "path": "minishlab/M2V_base_output"}, None)

        with self.assertRaises(ImportError):
            VectorsFactory.create({"method": "sentence-transformers", "path": "sentence-transformers/nli-mpnet-base-v2"}, None)

        with self.assertRaises(ImportError):
            VectorsFactory.create({"method": "words"}, None)

        # Test default model
        model = VectorsFactory.create({"path": "sentence-transformers/all-MiniLM-L6-v2"}, None)
        self.assertIsNotNone(model)

    def testWorkflow(self):
        """
        Test missing workflow dependencies
        """

        from txtai.workflow import ExportTask, ImageTask, ServiceTask, StorageTask, Workflow

        with self.assertRaises(ImportError):
            ExportTask()

        with self.assertRaises(ImportError):
            ImageTask()

        with self.assertRaises(ImportError):
            ServiceTask()

        with self.assertRaises(ImportError):
            StorageTask()

        with self.assertRaises(ImportError):
            Workflow([], workers=1).schedule(None, [])

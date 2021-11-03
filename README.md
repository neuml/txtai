<p align="center">
    <img src="https://raw.githubusercontent.com/neuml/txtai/master/logo.png"/>
</p>

<h3 align="center">
    <p>Build AI-powered semantic search applications</p>
</h3>

<p align="center">
    <a href="https://github.com/neuml/txtai/releases">
        <img src="https://img.shields.io/github/release/neuml/txtai.svg?style=flat&color=success" alt="Version"/>
    </a>
    <a href="https://github.com/neuml/txtai/releases">
        <img src="https://img.shields.io/github/release-date/neuml/txtai.svg?style=flat&color=blue" alt="GitHub Release Date"/>
    </a>
    <a href="https://github.com/neuml/txtai/issues">
        <img src="https://img.shields.io/github/issues/neuml/txtai.svg?style=flat&color=success" alt="GitHub issues"/>
    </a>
    <a href="https://github.com/neuml/txtai">
        <img src="https://img.shields.io/github/last-commit/neuml/txtai.svg?style=flat&color=blue" alt="GitHub last commit"/>
    </a>
    <a href="https://github.com/neuml/txtai/actions?query=workflow%3Abuild">
        <img src="https://github.com/neuml/txtai/workflows/build/badge.svg" alt="Build Status"/>
    </a>
    <a href="https://coveralls.io/github/neuml/txtai?branch=master">
        <img src="https://img.shields.io/coveralls/github/neuml/txtai" alt="Coverage Status">
    </a>
</p>

-------------------------------------------------------------------------------------------------------------------------------------------------------

txtai executes machine-learning workflows to transform data and build AI-powered semantic search applications.

![demo](https://raw.githubusercontent.com/neuml/txtai/master/demo.gif)

Traditional search systems use keywords to find data. Semantic search applications have an understanding of natural language and identify results that have the same meaning, not necessarily the same keywords.

Backed by state-of-the-art machine learning models, data is transformed into vector representations for search (also known as embeddings). Innovation is happening at a rapid pace, models can understand concepts in documents, audio, images and more.

Summary of txtai features:

- 🔎 Large-scale similarity search with multiple index backends ([Faiss](https://github.com/facebookresearch/faiss), [Annoy](https://github.com/spotify/annoy), [Hnswlib](https://github.com/nmslib/hnswlib))
- 📄 Create embeddings for text snippets, documents, audio, images and video. Supports transformers and word vectors.
- 💡 Machine-learning pipelines to run extractive question-answering, zero-shot labeling, transcription, translation, summarization and text extraction
- ↪️️ Workflows that join pipelines together to aggregate business logic. txtai processes can be microservices or full-fledged indexing workflows.
- 🔗 API bindings for [JavaScript](https://github.com/neuml/txtai.js), [Java](https://github.com/neuml/txtai.java), [Rust](https://github.com/neuml/txtai.rs) and [Go](https://github.com/neuml/txtai.go)
- ☁️ Cloud-native architecture that scales out with container orchestration systems (e.g. Kubernetes)

Applications range from similarity search to complex NLP-driven data extractions to generate structured databases. The following applications are powered by txtai.

| Application  | Description  |
|:----------|:-------------|
| [paperai](https://github.com/neuml/paperai) | AI-powered literature discovery and review engine for medical/scientific papers |
| [tldrstory](https://github.com/neuml/tldrstory) | AI-powered understanding of headlines and story text |
| [neuspo](https://neuspo.com) | Fact-driven, real-time sports event and news site |
| [codequestion](https://github.com/neuml/codequestion) | Ask coding questions directly from the terminal |

txtai is built with Python 3.6+, [Hugging Face Transformers](https://github.com/huggingface/transformers), [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and [FastAPI](https://github.com/tiangolo/fastapi)

## Why txtai?

In addition to traditional search systems, a growing number of semantic search solutions are available, so why txtai?

- `pip install txtai` is all that is needed to get started
- Works well with both small and big data, prototypes can be built in a couple lines of code, scale up as needed
- Rich data processing framework (pipelines and workflows) to pre and post process data
- Work in your programming language of choice via the API
- Modular with low install footprint, most dependencies optional and only required when you need them
- Learn by example, notebooks cover all available functionality

## Installation

The easiest way to install is via pip and PyPI

    pip install txtai

Python 3.6+ is supported. Using a Python [virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

See the detailed [install instructions](https://neuml.github.io/txtai/install) for more information covering
[installing from source](https://neuml.github.io/txtai/install/#install-from-source),
[environment specific prerequisites](https://neuml.github.io/txtai/install/#environment-specific-prerequisites) and
[optional dependencies](https://neuml.github.io/txtai/install/#optional-dependencies).

## Examples

The examples directory has a series of notebooks and applications giving an overview of txtai. See the sections below.

### Semantic Search

Build semantic/similarity/vector search applications.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Introducing txtai](https://github.com/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) | Overview of the functionality provided by txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) |
| [Build an Embeddings index with Hugging Face Datasets](https://github.com/neuml/txtai/blob/master/examples/02_Build_an_Embeddings_index_with_Hugging_Face_Datasets.ipynb) | Index and search Hugging Face Datasets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/02_Build_an_Embeddings_index_with_Hugging_Face_Datasets.ipynb) |
| [Build an Embeddings index from a data source](https://github.com/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb)  | Index and search a data source with word embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb) |
| [Add semantic search to Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb)  | Add semantic search to existing search systems | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb) |
| [API Gallery](https://github.com/neuml/txtai/blob/master/examples/08_API_Gallery.ipynb) | Using txtai in JavaScript, Java, Rust and Go | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/08_API_Gallery.ipynb) |
| [Similarity search with images](https://github.com/neuml/txtai/blob/master/examples/13_Similarity_search_with_images.ipynb) | Embed images and text into the same space for search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/13_Similarity_search_with_images.ipynb) |
| [Distributed embeddings cluster](https://github.com/neuml/txtai/blob/master/examples/15_Distributed_embeddings_cluster.ipynb) | Distribute an embeddings index across multiple data nodes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/15_Distributed_embeddings_cluster.ipynb) |

### Pipelines and Workflows

NLP-backed data transformation pipelines and workflows.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Extractive QA with txtai](https://github.com/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) | Introduction to extractive question-answering with txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) |
| [Extractive QA with Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) | Run extractive question-answering queries with Elasticsearch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) |
| [Extractive QA to build structured data](https://github.com/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) | Build structured datasets using extractive question-answering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) |
| [Apply labels with zero shot classification](https://github.com/neuml/txtai/blob/master/examples/07_Apply_labels_with_zero_shot_classification.ipynb) | Use zero shot learning for labeling, classification and topic modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/07_Apply_labels_with_zero_shot_classification.ipynb) |
| [Building abstractive text summaries](https://github.com/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) | Run abstractive text summarization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) |
| [Extract text from documents](https://github.com/neuml/txtai/blob/master/examples/10_Extract_text_from_documents.ipynb) | Extract text from PDF, Office, HTML and more | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/10_Extract_text_from_documents.ipynb) |
| [Transcribe audio to text](https://github.com/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) | Convert audio files to text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) |
| [Translate text between languages](https://github.com/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) | Streamline machine translation and language detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) |
| [Run pipeline workflows](https://github.com/neuml/txtai/blob/master/examples/14_Run_pipeline_workflows.ipynb) | Simple yet powerful constructs to efficiently process data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/14_Run_pipeline_workflows.ipynb) |
| [Transform tabular data with composable workflows](https://github.com/neuml/txtai/blob/master/examples/22_Transform_tabular_data_with_composable_workflows.ipynb) | Transform, index and search tabular data with workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/22_Transform_tabular_data_with_composable_workflows.ipynb) |

### Model Training

Train NLP models.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Train a text labeler](https://github.com/neuml/txtai/blob/master/examples/16_Train_a_text_labeler.ipynb) | Build text sequence classification models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/16_Train_a_text_labeler.ipynb) |
| [Train without labels](https://github.com/neuml/txtai/blob/master/examples/17_Train_without_labels.ipynb) | Use zero-shot classifiers to train new models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/17_Train_without_labels.ipynb) |
| [Train a QA model](https://github.com/neuml/txtai/blob/master/examples/19_Train_a_QA_model.ipynb) | Build and fine-tune question-answering models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/19_Train_a_QA_model.ipynb) |
| [Export and run models with ONNX](https://github.com/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb) | Export models with ONNX, run natively in JavaScript, Java and Rust | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb) |
| [Export and run other machine learning models](https://github.com/neuml/txtai/blob/master/examples/21_Export_and_run_other_machine_learning_models.ipynb) | Export and run models from scikit-learn, PyTorch and more | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/21_Export_and_run_other_machine_learning_models.ipynb) |

### Applications

Series of example applications with txtai.

| Application  | Description  | 
|:----------|:-------------|
| [Demo query shell](https://github.com/neuml/txtai/blob/master/examples/demo.py) | Basic similarity search example. Used in the original txtai demo. |
| [Book search](https://github.com/neuml/txtai/blob/master/examples/books.py) | Book similarity search application. Index book descriptions and query using natural language statements. |
| [Image search](https://github.com/neuml/txtai/blob/master/examples/images.py) | Image similarity search application. Index a directory of images and run searches to identify images similar to the input query. |
| [Wiki search](https://github.com/neuml/txtai/blob/master/examples/wiki.py) | Wikipedia search application. Queries Wikipedia API and summarizes the top result. |
| [Workflow builder](https://github.com/neuml/txtai/blob/master/examples/workflows.py) | Build and execute txtai workflows. Connect summarization, text extraction, transcription, translation and similarity search pipelines together to run unified workflows. |

### Documentation

[Full documentation on txtai](https://neuml.github.io/txtai) including configuration settings for pipelines, workflows, indexing and the API.

### Contributing

For those who would like to contribute to txtai, please see [this guide](https://github.com/neuml/.github/blob/master/CONTRIBUTING.md).

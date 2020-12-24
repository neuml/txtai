# txtai: AI-powered search engine

[![Version](https://img.shields.io/github/release/neuml/txtai.svg?style=flat&color=success)](https://github.com/neuml/txtai/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/neuml/txtai.svg?style=flat&color=blue)](https://github.com/neuml/txtai/releases)
[![GitHub issues](https://img.shields.io/github/issues/neuml/txtai.svg?style=flat&color=success)](https://github.com/neuml/txtai/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/neuml/txtai.svg?style=flat&color=blue)](https://github.com/neuml/txtai)
[![Build Status](https://github.com/neuml/txtai/workflows/build/badge.svg)](https://github.com/neuml/txtai/actions?query=workflow%3Abuild)
[![Coverage Status](https://img.shields.io/coveralls/github/neuml/txtai)](https://coveralls.io/github/neuml/txtai?branch=master)

<p align="center">
    <img width="250" src="https://raw.githubusercontent.com/neuml/txtai/master/logo.png"/>
</p>

txtai builds an AI-powered index over sections of text. txtai supports building text indices to perform similarity searches and create extractive question-answering based systems. txtai also has functionality for zero-shot classification.

![demo](https://raw.githubusercontent.com/neuml/txtai/master/demo.gif)

NeuML uses txtai and/or the concepts behind it to power all of our Natural Language Processing (NLP) applications. Example applications:

- [paperai](https://github.com/neuml/paperai) - AI-powered literature discovery and review engine for medical/scientific papers
- [tldrstory](https://github.com/neuml/tldrstory) - AI-powered understanding of headlines and story text
- [neuspo](https://neuspo.com) - Fact-driven, real-time sports event and news site
- [codequestion](https://github.com/neuml/codequestion) - Ask coding questions directly from the terminal

txtai is built on the following stack:

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [transformers](https://github.com/huggingface/transformers)
- [faiss](https://github.com/facebookresearch/faiss)
- Python 3.6+

## Installation
The easiest way to install is via pip and PyPI

    pip install txtai

You can also install txtai directly from GitHub. Using a Python Virtual Environment is recommended.

    pip install git+https://github.com/neuml/txtai

Python 3.6+ is supported

### Troubleshooting
This project has dependencies that require compiling native code. Windows and macOS systems require the following additional steps. Most Linux environments will install without any additional steps.

#### Windows

- Install C++ Build Tools - https://visualstudio.microsoft.com/visual-cpp-build-tools/
- PyTorch now has Windows binaries on PyPI and should work with the standard install. But if issues arise, try running the install directly from PyTorch.

    ```
    pip install txtai -f https://download.pytorch.org/whl/torch_stable.html
    ```

    See [pytorch.org](https://pytorch.org) for more information.

#### macOS

- Run the following before installing

    ```
    brew install libomp
    ```

    See [this link](https://github.com/kyamagu/faiss-wheels#prerequisite) for more information.

See this [GitHub workflow file](https://github.com/neuml/txtai/blob/master/.github/workflows/build.yml) for an example of environment-dependent installation procedures.

## Examples

The examples directory has a series of examples and notebooks giving an overview of txtai. See the list of notebooks below.

### Notebooks

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [Introducing txtai](https://github.com/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) | Overview of the functionality provided by txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) |
| [Extractive QA with txtai](https://github.com/neuml/txtai/blob/master/examples/02_Extractive_QA_with_txtai.ipynb) | Extractive question-answering with txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/02_Extractive_QA_with_txtai.ipynb) |
| [Build an Embeddings index from a data source](https://github.com/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb)  | Embeddings index from a data source backed by word embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb) |
| [Extractive QA with Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/04_Extractive_QA_with_Elasticsearch.ipynb) | Extractive question-answering with Elasticsearch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/04_Extractive_QA_with_Elasticsearch.ipynb) |
| [Labeling with zero-shot classification](https://github.com/neuml/txtai/blob/master/examples/05_Labeling_with_zero_shot_classification.ipynb) | Labeling with zero-shot classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/05_Labeling_with_zero_shot_classification.ipynb) |
| [API Gallery](https://github.com/neuml/txtai/blob/master/examples/06_API_Gallery.ipynb) | Using txtai in JavaScript, Java, Rust and Go | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/06_API_Gallery.ipynb) |

## Configuration

The following section goes over available settings for Embeddings and Extractor instances.

### Embeddings

Embeddings parameters are set through the constructor. Examples below.

```python
# Transformers embeddings model
Embeddings({"method": "transformers",
            "path": "sentence-transformers/bert-base-nli-mean-tokens"})

# Word embeddings model
Embeddings({"path": vectors,
            "storevectors": True,
            "scoring": "bm25",
            "pca": 3,
            "quantize": True})
```

#### method
```yaml
method: transformers|words
```

Sets the sentence embeddings method to use. When set to _transformers_, the embeddings object builds sentence embeddings using the sentence transformers. Otherwise a word embeddings model is used. Defaults to words.

#### path
```yaml
path: string
```

Required field that sets the path for a vectors model. When method set to _transformers_, this must be a path to a Hugging Face transformers model. Otherwise,
it must be a path to a local word embeddings model.

#### storevectors
```yaml
storevectors: boolean
```

Enables copying of a vectors model set in path into the embeddings models output directory on save. This option enables a fully encapsulated index with no external file dependencies.

#### scoring
```yaml
scoring: bm25|tfidf|sif
```

For word embedding models, a scoring model allows building weighted averages of word vectors for a given sentence. Supports BM25, tf-idf and SIF (smooth inverse frequency) methods. If a scoring method is not provided, mean sentence embeddings are built.

#### pca
```yaml
pca: int
```

Removes _n_ principal components from generated sentence embeddings. When enabled, a TruncatedSVD model is built to help with dimensionality reduction. After pooling of vectors creates a single sentence embedding, this method is applied.

#### backend
```yaml
backend: annoy|faiss|hnsw
```

Approximate Nearest Neighbor (ANN) index backend for storing generated sentence embeddings. Defaults to Faiss for Linux/macOS and Annoy for Windows. Faiss currently is not supported on Windows.

#### quantize
```yaml
quantize: boolean
```

Enables quanitization of generated sentence embeddings. If the index backend supports it, sentence embeddings will be stored with 8-bit precision vs 32-bit.
Only Faiss currently supports quantization.

### Extractor

Extractor parameters are set as constructor arguments. Examples below.

```python
Extractor(embeddings, path, quantize)
```

#### embeddings
```yaml
embeddings: Embeddings object instance
```

Embeddings object instance. Used to query and find candidate text snippets to run the question-answer model against.

#### path
```yaml
path: string
```

Required path to a Hugging Face SQuAD fine-tuned model. Used to answer questions.

#### quantize
```yaml
quantize: boolean
```

Enables dynamic quantization of the Hugging Face model. This is a runtime setting and doesn't save space. It is used to improve the inference time performance of the QA model.


### Labels

Labels parameters are set as constructor arguments. Examples below.

```python
Labels()
Labels("roberta-large-mnli")
```

#### path
```yaml
path: string
```

Required path to a Hugging Face MNLI fine-tuned model. Used to answer questions.

### API

txtai has a full-featured API that can optionally be enabled for any txtai process. All functionality found in txtai can be accessed via the API. The following is an example configuration and startup script for the API.

Note that this configuration file enables all functionality (embeddings, extractor and labels). It is suggested that separate processes are used for each instance of a txtai component.

```yaml
# Index file path
path: /tmp/index

# Allow indexing of documents
writable: True

# Embeddings settings
embeddings:
  method: transformers
  path: sentence-transformers/bert-base-nli-mean-tokens

# Extractor settings
extractor:
  path: distilbert-base-cased-distilled-squad

# Labels settings
labels:
```

Assuming this YAML content is stored in a file named index.yml, the following command starts the API process.

```
CONFIG=index.yml uvicorn "txtai.api:app"
```

### Supported language bindings

The following programming languages have txtai bindings:

- [JavaScript](https://github.com/neuml/txtai.js)
- [Java](https://github.com/neuml/txtai.java)
- [Rust](https://github.com/neuml/txtai.rs)
- [Go](https://github.com/neuml/txtai.go)

For additional language bindings, please add an issue!

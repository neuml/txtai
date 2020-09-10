# txtai: AI-powered search engine

[![Version](https://img.shields.io/github/release/neuml/txtai.svg?style=flat&color=success)](https://github.com/neuml/txtai/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/neuml/txtai.svg?style=flat&color=blue)](https://github.com/neuml/txtai/releases)
[![GitHub issues](https://img.shields.io/github/issues/neuml/txtai.svg?style=flat&color=success)](https://github.com/neuml/txtai/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/neuml/txtai.svg?style=flat&color=blue)](https://github.com/neuml/txtai)
[![Build Status](https://img.shields.io/travis/neuml/txtai/master.svg?style=flat)](https://travis-ci.org/neuml/txtai)
[![Coverage Status](https://img.shields.io/coveralls/github/neuml/txtai)](https://coveralls.io/github/neuml/txtai?branch=master)

txtai builds an AI-powered index over sections of text. txtai supports building text indices to perform similarity searches and create extractive question-answering based systems. 

![demo](https://raw.githubusercontent.com/neuml/txtai/master/demo.gif)

NeuML uses txtai and/or the concepts behind it to power all of our Natural Language Processing (NLP) applications. Example applications:

- [cord19q](https://github.com/neuml/cord19q) - COVID-19 literature analysis
- [paperai](https://github.com/neuml/paperai) - AI-powered literature discovery and review engine for medical/scientific papers
- [neuspo](https://neuspo.com) - a fact-driven, real-time sports event and news site
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
- PyTorch Windows binaries are not on PyPI, the following url link must be added when installing

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

## Examples

The examples directory has a series of examples and notebooks giving an overview of txtai. See the list of notebooks below.

### Notebooks

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [Introducing txtai](https://github.com/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb)  | Overview of the functionality provided by txtai  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) |
| [Extractive QA with txtai](https://github.com/neuml/txtai/blob/master/examples/02_Extractive_QA_with_txtai.ipynb)  | Extractive question-answering with txtai  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/02_Extractive_QA_with_txtai.ipynb) |
| [Build an Embeddings index from a data source](https://github.com/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb)  | Embeddings index from a data source backed by word embeddings |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb) |
| [Extractive QA with Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/04_Extractive_QA_with_Elasticsearch.ipynb)  | Extractive question-answering with Elasticsearch  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/04_Extractive_QA_with_Elasticsearch.ipynb) |

## Configuration

The following section goes over available settings for Embeddings and Extractor instances.

### Embeddings

Embeddings methods are set through the constructor. Examples below.

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

Extractor methods are set as constructor arguments. Examples below.

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

<p align="center">
    <img src="https://raw.githubusercontent.com/neuml/txtai/master/logo.png"/>
</p>

<h3 align="center">
    <p>AI-powered search engine</p>
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
| [Build an Embeddings index with Hugging Face Datasets](https://github.com/neuml/txtai/blob/master/examples/02_Build_an_Embeddings_index_with_Hugging_Face_Datasets.ipynb) | Index and search Hugging Face Datasets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/02_Build_an_Embeddings_index_with_Hugging_Face_Datasets.ipynb) |
| [Build an Embeddings index from a data source](https://github.com/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb)  | Index and search a data source with word embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb) |
| [Add semantic search to Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb)  | Add semantic search to existing search systems | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb) |
| [Extractive QA with txtai](https://github.com/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) | Introduction to extractive question-answering with txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) |
| [Extractive QA with Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) | Run extractive question-answering queries with Elasticsearch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) |
| [Apply labels with zero shot classification](https://github.com/neuml/txtai/blob/master/examples/07_Apply_labels_with_zero_shot_classification.ipynb) | Use zero shot learning for labeling, classification and topic modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/07_Apply_labels_with_zero_shot_classification.ipynb) |
| [API Gallery](https://github.com/neuml/txtai/blob/master/examples/08_API_Gallery.ipynb) | Using txtai in JavaScript, Java, Rust and Go | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/08_API_Gallery.ipynb) |

## Configuration

The following sections cover available settings for each txtai component. See the example notebooks for detailed examples on how to use each txtai component.

### Embeddings

An Embeddings instance is the engine that provides similarity search. Embeddings can be used to run ad-hoc similarity comparisions or build/search large indices.

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

Backend-specific settings are set with a corresponding configuration object having the same name as the backend (i.e. annoy, faiss, or hnsw). None of these are required and are set to defaults if omitted.

#### annoy
```yaml
annoy:
  ntrees: number of trees (int) - defaults to 10
  searchk: search_k search setting (int) - defaults to -1
```

See [Annoy documentation](https://github.com/spotify/annoy#full-python-api) for more information on these parameters.

#### faiss
```yaml
faiss:
  components: Comma separated list of components - defaults to None
  nprobe: search probe setting (int) - defaults to 6
```

See Faiss documentation on the [index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory) and [search](https://github.com/facebookresearch/faiss/wiki/Faster-search) for more information on these parameters.

#### hnsw
```yaml
hnsw:
  efconstruction:  ef_construction param for init_index (int) - defaults to 200
  m: M param for init_index (int) - defaults to 16
  randomseed: random-seed param for init_index (init) - defaults to 100
  efsearch: ef search param (int) - defaults to None and not set
```

See [Hnswlib documentation](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) for more information on these parameters.

#### quantize
```yaml
quantize: boolean
```

Enables quanitization of generated sentence embeddings. If the index backend supports it, sentence embeddings will be stored with 8-bit precision vs 32-bit.
Only Faiss currently supports quantization.

### Pipelines

txtai provides a light wrapper around a couple of the Hugging Face pipelines. All pipelines have the following common parameters.

#### path
```yaml
path: string
```

Required path to a Hugging Face model

#### quantize
```yaml
quantize: boolean
```

Enables dynamic quantization of the Hugging Face model. This is a runtime setting and doesn't save space. It is used to improve the inference time performance of models.

#### gpu
```yaml
gpu: boolean
```

Enables GPU inference.

#### model
```yaml
model: Hugging Face pipeline or txtai pipeline
```

Shares the underlying model of the passed in pipeline with this pipeline. This allows having variations of a pipeline without having to store multiple copies of the full model in memory.

### Extractor

An Extractor pipeline is a combination of an embeddings query and an Extractive QA model. Filtering the context for a QA model helps maximize performance of the model.

Extractor parameters are set as constructor arguments. Examples below.

```python
Extractor(embeddings, path, quantize, gpu, model, tokenizer)
```

#### embeddings
```yaml
embeddings: Embeddings object instance
```

Embeddings object instance. Used to query and find candidate text snippets to run the question-answer model against.

#### tokenizer
```yaml
tokenizer: Tokenizer function
```

Optional custom tokenizer function to parse input queries

### Labels

A Labels pipeline uses a zero shot classification model to apply labels to input text. 

Labels parameters are set as constructor arguments. Examples below.

```python
Labels()
Labels("roberta-large-mnli")
```

### Similarity

A Similarity pipeline is also a zero shot classifier model where the labels are the queries. The results are transposed to get scores per query/label vs scores per input text. 

Similarity parameters are set as constructor arguments. Examples below.

```python
Similarity()
Similarity("roberta-large-mnli")
```

### API

txtai has a full-featured API that can optionally be enabled for any txtai process. All functionality found in txtai can be accessed via the API. The following is an example configuration and startup script for the API.

Note that this configuration file enables all functionality (embeddings, extractor, labels, similarity). It is suggested that separate processes are used for each instance of a txtai component.

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

# Similarity settings
similarity:
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

# API

![api](../images/api.png#only-light)
![api](../images/api-dark.png#only-dark)

txtai has a full-featured API, backed by [FastAPI](https://github.com/tiangolo/fastapi), that can optionally be enabled for any txtai process. All functionality found in txtai can be accessed via the API.

The following is an example configuration and startup script for the API.

Note: that this configuration file enables all functionality. It is suggested that separate processes are used for each instance of a txtai component.

```yaml
# Index file path
path: /tmp/index

# Allow indexing of documents
writable: True

# Enbeddings index
embeddings:
  path: sentence-transformers/nli-mpnet-base-v2

# Extractive QA
extractor:
  path: distilbert-base-cased-distilled-squad

# Zero-shot labeling
labels:

# Similarity
similarity:

# Text segmentation
segmentation:
    sentences: true

# Text summarization
summary:

# Text extraction
textractor:
    paragraphs: true
    minlength: 100
    join: true

# Transcribe audio to text
transcription:

# Translate text between languages
translation:

# Workflow definitions
workflow:
    sumfrench:
        tasks:
            - action: textractor
              task: storage
              ids: false
            - action: summary
            - action: translation
              args: ["fr"]
    sumspanish:
        tasks:
            - action: textractor
              task: url
            - action: summary
            - action: translation
              args: ["es"]
```

Assuming this YAML content is stored in a file named index.yml, the following command starts the API process.

```bash
CONFIG=index.yml uvicorn "txtai.api:app"
```

uvicorn is a full-featured production ready server with support for SSL and more. See the [uvicorn deployment guide](https://www.uvicorn.org/deployment/) for details.

## Connect to API

The default port for the API is 8000. See the uvicorn link above to change this.

txtai has a number of language bindings which abstract the API (see links below). Alternatively, code can be written to connect directly to the API. Documentation for a live running instance can be found at the `/docs` url (i.e. http://localhost:8000/docs). The following example runs a workflow using cURL.

```bash
curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"sumfrench", "elements": ["https://github.com/neuml/txtai"]}'
```

## Supported language bindings

The following programming languages have txtai bindings:

- [JavaScript](https://github.com/neuml/txtai.js)
- [Java](https://github.com/neuml/txtai.java)
- [Rust](https://github.com/neuml/txtai.rs)
- [Go](https://github.com/neuml/txtai.go)

See the link below for a detailed example covering how to use the API.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [API Gallery](https://github.com/neuml/txtai/blob/master/examples/08_API_Gallery.ipynb) | Using txtai in JavaScript, Java, Rust and Go | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/08_API_Gallery.ipynb) |

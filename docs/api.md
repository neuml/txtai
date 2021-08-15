# API

txtai has a full-featured API that can optionally be enabled for any txtai process. All functionality found in txtai can be accessed via the API. The following is an example configuration and startup script for the API.

Note that this configuration file enables all functionality. It is suggested that separate processes are used for each instance of a txtai component. Components can be joined together with workflows.

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

```
CONFIG=index.yml uvicorn "txtai.api:app"
```

uvicorn is a full-featured production ready server with support for SSL and more. See the [uvicorn deployment guide](https://www.uvicorn.org/deployment/) for details.

## Docker

A Dockerfile with commands to install txtai, all dependencies and default configuration is available in this repository.

The Dockerfile can be copied from the docker directory on GitHub locally. The following commands show how to run the API process.

```bash
docker build -t txtai.api -f docker/api.Dockerfile .
docker run --name txtai.api -p 8000:8000 --rm -it txtai.api

# Alternatively, if nvidia-docker is installed, the build will support GPU runtimes
docker run --name txtai.api --runtime=nvidia -p 8000:8000 --rm -it txtai.api
```

This will bring up an API instance without having to install Python, txtai or any dependencies on your machine!

## Distributed embeddings clusters

The API supports combining multiple API instances into a single logical embeddings index. An example configuration is shown below.

```yaml
cluster:
    shards:
        - http://127.0.0.1:8002
        - http://127.0.0.1:8003
```

This configuration aggregates the API instances above as index shards. Data is evenly split among each of the shards at index time. Queries are run in parallel against each shard and the results are joined together. This method allows horizontal scaling and supports very large index clusters.

This method is only recommended for data sets in the 1 billion+ records. The ANN libraries can easily support smaller data sizes and this method is not worth the additional complexity. At this time, new shards can not be added after building the initial index.

## Differences between Python and API

The txtai API provides all the major functionality found in this project. But there are differences due to the nature of JSON and differences across the supported programming languages. For example, any Python callable method is available at a named endpoint (i.e. instead of summary() the method call would be summary.summary()).
Return types vary as tuples are returned as objects via the API.

## Supported language bindings

The following programming languages have txtai bindings:

- [JavaScript](https://github.com/neuml/txtai.js)
- [Java](https://github.com/neuml/txtai.java)
- [Rust](https://github.com/neuml/txtai.rs)
- [Go](https://github.com/neuml/txtai.go)

See each of the projects above for details on how to install and use. Please add an issue to request additional language bindings!

::: txtai.api.application
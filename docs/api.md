# API

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

## Differences between Python and API

The txtai API provides all the major functionality found in this project. But there are differences due to the nature of JSON and differences across the supported programming languages.

|  Difference  |    Python    |  API  |  Reason  |
|:-------------|:-------------|:------|:---------|
| Return Types | tuples | objects | Consistency across languages. For example, (id, score) in Python is {"id": value, "score": value} via API |
| Extractor    | extract() | extractor.extract() | Extractor pipeline is a callable object in Python |
| Labels       | labels()  | labels.label() | Labels pipeline is a callable object in Python that supports both string and list input |
| Similarity   | similarity() | similarity.similarity() | Similarity pipeline a callable object in Python that supports both string and list input |

## Supported language bindings

The following programming languages have txtai bindings:

- [JavaScript](https://github.com/neuml/txtai.js)
- [Java](https://github.com/neuml/txtai.java)
- [Rust](https://github.com/neuml/txtai.rs)
- [Go](https://github.com/neuml/txtai.go)

See each of the projects above for details on how to install and use. Please add an issue to request additional language bindings!

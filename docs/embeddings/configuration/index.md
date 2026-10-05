# Configuration

The following describes available embeddings configuration. These parameters are set in the [Embeddings constructor](../methods#txtai.embeddings.base.Embeddings.__init__) via either the `config` parameter or as keyword arguments.

Configuration is designed to be optional and set only when needed. Out of the box, sensible defaults are picked to get up and running fast. For example:

```python
from txtai import Embeddings

embeddings = Embeddings()
```

Creates a new embeddings instance, using [all-MiniLM-L6-v2](https://hf.co/sentence-transformers/all-MiniLM-L6-v2) as the vector model, [Faiss](https://faiss.ai/) as the ANN index backend and content disabled.

```python
from txtai import Embeddings

embeddings = Embeddings(content=True)
```

Is the same as above except it adds in [SQLite](https://www.sqlite.org/index.html) for content storage. 

The following sections link to all the available configuration options.

## [ANN](./ann)

The default vector index backend is Faiss.

## [Cloud](./cloud)

Embeddings databases can optionally be synced with cloud storage.

## [Database](./database)

Content storage is disabled by default. When enabled, SQLite is the default storage engine.

## [General](./general)

General configuration that doesn't fit elsewhere.

## [Graph](./graph)

An accomplying graph index can be created with an embeddings database. This enables topic modeling, path traversal and more. [NetworkX](https://github.com/networkx/networkx) is the default graph index.

## [Scoring](./scoring)

Sparse keyword indexing and word vectors term weighting.

## [Vectors](./vectors)

Vector search is enabled by converting text and other binary data into embeddings vectors. These vectors are then stored in an ANN index. The vector model is optional and a default model is used when not provided.

# Index format

![format](../images/format.png#only-light)
![format](../images/format-dark.png#only-dark)

This section documents the txtai index format. Each component is designed to ensure open access to the underlying data in a programmatic and platform independent way

If an underlying library has an index format, that is used. Otherwise, txtai persists content with [MessagePack](https://msgpack.org/index.html) serialization.

To learn more about how these components work together, read the [Index Guide](../indexing) and [Query Guide](../query).

## ANN

Approximate Nearest Neighbor (ANN) index configuration for storing vector embeddings.

| Component                                                     | Storage Format                                                               |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [Faiss](https://github.com/facebookresearch/faiss)            | Local file format provided by library                                        |
| [Hnswlib](https://github.com/nmslib/hnswlib)                  | Local file format provided by library                                        |
| [Annoy](https://github.com/spotify/annoy)                     | Local file format provided by library                                        |
| [NumPy](https://github.com/numpy/numpy)                       | Local NumPy array files via np.save / np.load                                |
| [Postgres via pgvector](https://github.com/pgvector/pgvector) | Vector tables in a Postgres database                                         |

## Core

Core embeddings index files.

| Component                                                     | Storage Format                                                               |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [Configuration](https://www.json.org/)                        | Embeddings index configuration stored as JSON                                |
| [Index Ids](https://msgpack.org/index.html)                   | Embeddings index ids serialized with MessagePack. Only enabled when when content storage (database) is disabled. |

## Database

Databases store metadata, text and binary content.

| Component                                                     | Storage Format                                                               |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [SQLite](https://www.sqlite.org/)                             | Local database files with SQLite                                             |
| [DuckDB](https://github.com/duckdb/duckdb)                    | Local database files with DuckDB                                             |
| [Postgres](https://www.postgresql.org/)                       | Postgres relational database via [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy). Supports additional databases via this library. |

## Graph

Graph nodes and edges for an embeddings index

| Component                                                     | Storage Format                                                                |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [NetworkX](https://github.com/networkx/networkx)              | Nodes and edges exported to local file serialized with MessagePack            |
| [Postgres](https://github.com/aplbrain/grand)                 | Nodes and edges stored in a Postgres database. Supports additional databases. |

## Scoring

Sparse/keyword indexing

| Component                                                     | Storage Format                                                                |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [Local index](https://www.sqlite.org/)                        | Metadata serialized with MessagePack. Terms stored in SQLite.                 |
| [Postgres](https://www.postgresql.org/docs/current/textsearch.html) | Text indexed with Postgres Full Text Search (FTS)                             |

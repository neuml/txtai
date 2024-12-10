# ANN

Approximate Nearest Neighbor (ANN) index configuration for storing vector embeddings.

## backend
```yaml
backend: faiss|hnsw|annoy|numpy|torch|pgvector|sqlite|custom
```

Sets the ANN backend. Defaults to `faiss`. Additional backends are available via the [ann](../../../install/#ann) extras package. Set custom backends via setting this parameter to the fully resolvable class string.

Backend-specific settings are set with a corresponding configuration object having the same name as the backend (i.e. annoy, faiss, or hnsw). These are optional and set to defaults if omitted.

### faiss
```yaml
faiss:
    components: comma separated list of components - defaults to "IDMap,Flat" for small
                indices and "IVFx,Flat" for larger indexes where
                x = min(4 * sqrt(embeddings count), embeddings count / 39)
                automatically calculates number of IVF cells when omitted (supports "IVF,Flat")
    nprobe: search probe setting (int) - defaults to x/16 (as defined above)
            for larger indexes
    nflip: same as nprobe - only used with binary hash indexes
    quantize: store vectors with x-bit precision vs 32-bit (boolean|int)
              true sets 8-bit precision, false disables, int sets specified
              precision
    mmap: load as on-disk index (boolean) - trade query response time for a
          smaller RAM footprint, defaults to false
    sample: percent of data to use for model training (0.0 - 1.0)
            reduces indexing time for larger (>1M+ row) indexes, defaults to 1.0
```

Faiss supports both floating point and binary indexes. Floating point indexes are the default. Binary indexes are used when indexing scalar-quantized datasets.

See the following Faiss documentation links for more information.

- [Guidelines for choosing an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [Index configuration summary](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Index Factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
- [Binary Indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes)
- [Search Tuning](https://github.com/facebookresearch/faiss/wiki/Faster-search)

Note: For macOS users, an existing bug in an upstream package restricts the number of processing threads to 1. This limitation is managed internally to prevent system crashes.

### hnsw
```yaml
hnsw:
    efconstruction:  ef_construction param for init_index (int) - defaults to 200
    m: M param for init_index (int) - defaults to 16
    randomseed: random-seed param for init_index (int) - defaults to 100
    efsearch: ef search param (int) - defaults to None and not set
```

See [Hnswlib documentation](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) for more information on these parameters.

### annoy
```yaml
annoy:
    ntrees: number of trees (int) - defaults to 10
    searchk: search_k search setting (int) - defaults to -1
```

See [Annoy documentation](https://github.com/spotify/annoy#full-python-api) for more information on these parameters. Note that annoy indexes can not be modified after creation, upserts/deletes and other modifications are not supported.

### numpy

The NumPy backend is a k-nearest neighbors backend. It's designed for simplicity and works well with smaller datasets.

The `torch` backend supports the same options. The only difference is that the vectors can be search using GPUs.

### pgvector
```yaml
pgvector:
    url: database url connection string, alternatively can be set via
         ANN_URL environment variable
    schema: database schema to store vectors - defaults to being
            determined by the database
    table: database table to store vectors - defaults to `vectors`
    precision: vector float precision (half or full) - defaults to `full`
    efconstruction:  ef_construction param (int) - defaults to 200
    m: M param for init_index (int) - defaults to 16
```

The pgvector backend stores embeddings in a Postgres database. See the [pgvector documentation](https://github.com/pgvector/pgvector-python?tab=readme-ov-file#sqlalchemy) for more information on these parameters. See the [SQLAlchemy](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) documentation for more information on how to construct url connection strings.

### sqlite
```yaml
sqlite:
    quantize: store vectors with x-bit precision vs 32-bit (boolean|int)
              true sets 8-bit precision, false disables, int sets specified
              precision
    table: database table to store vectors - defaults to `vectors`
```

The SQLite backend stores embeddings in a SQLite database using [sqlite-vec](https://github.com/asg017/sqlite-vec). This backend supports 1-bit and 8-bit quantization at the storage level.

See [this note](https://alexgarcia.xyz/sqlite-vec/python.html#macos-blocks-sqlite-extensions-by-default) on how to run this ANN on MacOS.

# ANN

The following covers the available vector index configuration options.

## backend
```yaml
backend: faiss|hnsw|annoy|numpy|torch|custom
```

Approximate Nearest Neighbor (ANN) index backend for storing generated sentence embeddings. `Defaults to faiss`. Additional backends require the
[similarity](../../../install/#similarity) extras package to be installed. Add custom backends via setting this parameter to the fully resolvable
class string.

Backend-specific settings are set with a corresponding configuration object having the same name as the backend (i.e. annoy, faiss, or hnsw). None of these are required and are set to defaults if omitted.

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
    quantize: store vectors with x-bit precision vs 32-bit (bool|int)
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

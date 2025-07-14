# Scoring

Enable scoring support via the `scoring` parameter.

This scoring instance can serve two purposes, depending on the settings.

One use case is building sparse/keyword indexes. This occurs when the `terms` parameter is set to `True`.

The other use case is with word vector term weighting. This feature has been available since the initial version but isn't quite as common anymore.

The following covers the available options.

## method
```yaml
method: bm25|tfidf|sif|pgtext|sparse|custom
```

Sets the scoring method. Add custom scoring via setting this parameter to the fully resolvable class string.

### pgtext
```yaml
schema: database schema to store keyword index - defaults to being
        determined by the database
```

Additional settings for Postgres full-text keyword indexes.

### sparse
```yaml
path: sparse vector model path
vectormethod: vector embeddings method
gpu: boolean|int|string|device
batch: Sets the transform batch size
encodebatch: Sets the encode batch size
vectors: additional model init args
encodeargs: additional encode() args
backend: ivfsparse|pgsparse
```

Sparse vector scoring options. The sparse scoring instance combines a sparse vector model with a sparse approximate nearest neighbor index (ANN).

#### ivfsparse
```yaml
ivfsparse:
  sample: percent of data to use for model training (0.0 - 1.0)
  nlist: desired number of clusters (int)
  nprobe: search probe setting (int)
```

Inverted file (IVF) index with flat vector file storage and sparse array support.

#### pgsparse

Sparse ANN backed by Postgres. Supports same options as the [pgvector](../ann/#pgvector) ANN.

## terms
```yaml
terms: boolean|dict
```

Enables term frequency sparse arrays for a scoring instance. This is the backend for sparse keyword indexes.

Supports a `dict` with the parameters `cachelimit` and `cutoff`.

`cachelimit` is the maximum amount of resident memory in bytes to use during indexing before flushing to disk. This parameter is an `int`.

`cutoff` is used during search to determine what constitutes a common term. This parameter is a `float`, i.e. 0.1 for a cutoff of 10%.

When `terms` is set to `True`, default parameters are used for the `cachelimit` and `cutoff`. Normally, these defaults are sufficient.

## normalize
```yaml
normalize: boolean
```

Enables normalized scoring (ranging from 0 to 1). When enabled, statistics from the index will be used to calculate normalized scores.

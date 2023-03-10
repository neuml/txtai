# Configuration

This following describes available embeddings configuration. These parameters are set via the [Embeddings constructor](../methods#txtai.embeddings.base.Embeddings.__init__).

## format
```yaml
format: pickle|json
```

Sets the configuration storage format. Defaults to pickle.

## path
```yaml
path: string
```

Sets the path for a vectors model. When using a transformers/sentence-transformers model, this can be any model on the
[Hugging Face Hub](https://huggingface.co/models) or a local file path. Otherwise, it must be a local file path to a word embeddings model.

## method
```yaml
method: transformers|sentence-transformers|words|external
```

Sentence embeddings method to use. If the method is not provided, it is inferred using the `path`.

`sentence-transformers` and `words` require the [similarity](../../install/#similarity) extras package to be installed.

### transformers

Builds sentence embeddings using a transformers model. While this can be any transformers model, it works best with
[models trained](https://huggingface.co/models?pipeline_tag=sentence-similarity) to build sentence embeddings.

### sentence-transformers

Same as transformers but loads models with the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library.

### words

Builds sentence embeddings using a word embeddings model. Transformers models are the preferred vector backend in most cases. Word embeddings models may be deprecated in the future.

#### storevectors
```yaml
storevectors: boolean
```

Enables copying of a vectors model set in path into the embeddings models output directory on save. This option enables a fully encapsulated index with no external file dependencies.

#### scoring
```yaml
scoring: bm25|tfidf|sif
```

A scoring model builds weighted averages of word vectors for a given sentence. Supports BM25, TF-IDF and SIF (smooth inverse frequency) methods. If a scoring method is not provided, mean sentence embeddings are built.

#### pca
```yaml
pca: int
```

Removes _n_ principal components from generated sentence embeddings. When enabled, a TruncatedSVD model is built to help with dimensionality reduction. After pooling of vectors creates a single sentence embedding, this method is applied.

### external

Sentence embeddings are loaded via an external model or API. Requires setting the [transform](#transform) parameter to a function that translates data into vectors.

#### transform
```yaml
transform: function
```

When method is `external`, this function transforms input content into embeddings. The input to this function is a list of data. This method must return either a numpy array or list of numpy arrays.

## batch
```yaml
batch: int
```

Sets the transform batch size. This parameter controls how input streams are chunked and vectorized.

## encodebatch
```yaml
encodebatch: int
```

Sets the encode batch size. This parameter controls the underlying vector model batch size. This often corresponds to a GPU batch size, which controls GPU memory usage.

## tokenize
```yaml
tokenize: boolean
```

Enables string tokenization (defaults to false). This method applies tokenization rules that only work with English language text and may increase the quality of
English language sentence embeddings in some situations.

## instructions
```yaml
instructions:
    query: prefix for queries
    data: prefix for indexing
```

Instruction-based models use prefixes to modify how embeddings are computed. This is especially useful with asymmetric search, which is when the query and indexed data are of vastly different lengths. In other words, short queries with long documents.

[E5-base](https://huggingface.co/intfloat/e5-base) is an example of a model that accepts instructions. It takes `query: ` and `passage: ` prefixes and uses those to generate embeddings that work well for asymmetric search.

## backend
```yaml
backend: faiss|hnsw|annoy
```

Approximate Nearest Neighbor (ANN) index backend for storing generated sentence embeddings. `Defaults to faiss`. Additional backends require the
[similarity](../../install/#similarity) extras package to be installed.

Backend-specific settings are set with a corresponding configuration object having the same name as the backend (i.e. annoy, faiss, or hnsw). None of these are required and are set to defaults if omitted.

### faiss
```yaml
faiss:
    components: comma separated list of components - defaults to "Flat" for small
                indices and "IVFx,Flat" for larger indexes where
                x = 4 * sqrt(embeddings count)
    nprobe: search probe setting (int) - defaults to x/16 (as defined above)
            for larger indexes
    quantize: store vectors with 8-bit precision vs 32-bit (boolean)
              defaults to false
    mmap: load as on-disk index (boolean) - trade query response time for a
          smaller RAM footprint, defaults to false
    sample: percent of data to use for model training (0.0 - 1.0)
            reduces indexing time for larger (>1M+ row) indexes, defaults to 1.0
```

See the following Faiss documentation links for more information.

- [Guidelines for choosing an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [Index configuration summary](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Index Factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
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

## content
```yaml
content: string|boolean
```

Enables content storage. When true, the default content storage engine will be used. `Defaults to sqlite`. Otherwise, the string must specify the supported content storage engine to use.

## functions
```yaml
functions: list
```

List of functions with user-defined SQL functions, only used when [content](#content) is enabled. Each list element must be one of the following:

- function
- callable object
- dict with fields for name, argcount and function

[An example can be found here](../query#custom-sql-functions).

## query
```yaml
query:
    path: sets the path for the query model - this can be any model on the
          Hugging Face Model Hub or a local file path.
    prefix: text prefix to prepend to all inputs
    maxlength: maximum generated sequence length
```

Query translation model. Translates natural language queries to txtai compatible SQL statements.

## graph
```yaml
graph:
    backend: graph network backend (string), defaults to "networkx"
    batchsize: batch query size, used to query embeddings index (int)
               defaults to 256
    limit: maximum number of results to return per embeddings query (int)
           defaults to 15
    minscore: minimum score required to consider embeddings query matches (float)
              defaults to 0.1
    approximate: when true, queries only run for nodes without edges (boolean)
                 defaults to true
    topics: see below
```

Enables graph storage. When set, a graph network is built using the embeddings index. Graph nodes are synced with each embeddings index operation (index/upsert/delete). Graph edges are created using the embeddings index upon completion of each index/upsert/delete embeddings index call.

Defaults are tuned so that in most cases these values don't need to be changed. 

### topics
```yaml
topics:
    algorithm: community detection algorithm (string), options are
               louvain (default), greedy, lpa
    level: controls number of topics (string), options are best (default) or first
    resolution: controls number of topics (int), larger values create more
                topics (int), defaults to 100
    labels: scoring index method used to build topic labels (string)
            options are bm25 (default), tfidf, sif
    terms: number of frequent terms to use for topic labels (int), defaults to 4
    stopwords: optional list of stop words to exclude from topic labels
    categories: optional list of categories used to group topics, allows
                granular topics with broad categories grouping topics
```

Enables topic modeling. Defaults are tuned so that in most cases these values don't need to be changed (except for categories). These parameters are available for advanced use cases where one wants full control over the community detection process.

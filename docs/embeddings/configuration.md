# Configuration

## Embeddings
This following describes available embeddings configuration. These parameters are set via the [Embeddings constructor](../methods#txtai.embeddings.base.Embeddings.__init__).

### path
```yaml
path: string
```

Sets the path for a vectors model. When using a transformers/sentence-transformers model, this can be any model on the
[Hugging Face Model Hub](https://huggingface.co/models) or a local file path. Otherwise, it must be a local file path to a word embeddings model.

### method
```yaml
method: transformers|sentence-transformers|words|external
```

Sentence embeddings method to use. If the method is not provided, it is inferred using the `path`.

`sentence-transformers` and `words` require the [similarity](../../install/#similarity) extras package to be installed.

#### transformers

Builds sentence embeddings using a transformers model. While this can be any transformers model, it works best with
[models trained](https://huggingface.co/models?pipeline_tag=sentence-similarity) to build sentence embeddings.

#### sentence-transformers

Same as transformers but loads models with the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library.

#### words

Builds sentence embeddings using a word embeddings model. Transformers models are the preferred vector backend in most cases. Word embeddings models may be deprecated in the future.

##### storevectors
```yaml
storevectors: boolean
```

Enables copying of a vectors model set in path into the embeddings models output directory on save. This option enables a fully encapsulated index with no external file dependencies.

##### scoring
```yaml
scoring: bm25|tfidf|sif
```

A scoring model builds weighted averages of word vectors for a given sentence. Supports BM25, TF-IDF and SIF (smooth inverse frequency) methods. If a scoring method is not provided, mean sentence embeddings are built.

##### pca
```yaml
pca: int
```

Removes _n_ principal components from generated sentence embeddings. When enabled, a TruncatedSVD model is built to help with dimensionality reduction. After pooling of vectors creates a single sentence embedding, this method is applied.

#### external

Sentence embeddings are loaded via an external model or API. Requires setting the [transform](#transform) parameter to a function that translates data into vectors.

##### transform
```yaml
transform: function
```

When method is `external`, this function transforms input content into embeddings.

### backend
```yaml
backend: faiss|hnsw|annoy
```

Approximate Nearest Neighbor (ANN) index backend for storing generated sentence embeddings. `Defaults to Faiss`. Additional backends require the
[similarity](../../install/#similarity) extras package to be installed.

Backend-specific settings are set with a corresponding configuration object having the same name as the backend (i.e. annoy, faiss, or hnsw). None of these are required and are set to defaults if omitted.

#### faiss
```yaml
faiss:
    components: Comma separated list of components - defaults to "Flat" for small
                indices and "IVFx,Flat" for larger indexes where
                x = 4 * sqrt(embeddings count)
    nprobe: search probe setting (int) - defaults to x/16 (as defined above)
            for larger indexes
```

See the following Faiss documentation links for more information.

- [Guidelines for choosing an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [Index configuration summary](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Index Factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
- [Search Tuning](https://github.com/facebookresearch/faiss/wiki/Faster-search)

#### hnsw
```yaml
hnsw:
    efconstruction:  ef_construction param for init_index (int) - defaults to 200
    m: M param for init_index (int) - defaults to 16
    randomseed: random-seed param for init_index (init) - defaults to 100
    efsearch: ef search param (int) - defaults to None and not set
```

See [Hnswlib documentation](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) for more information on these parameters.

#### annoy
```yaml
annoy:
    ntrees: number of trees (int) - defaults to 10
    searchk: search_k search setting (int) - defaults to -1
```

See [Annoy documentation](https://github.com/spotify/annoy#full-python-api) for more information on these parameters. Note that annoy indexes can not be modified after creation, upserts/deletes and other modifications are not supported.

### content
```yaml
content: string|boolean
```

Enables content storage. When true, the default content storage engine will be used. Otherwise, the string must specify the supported content storage engine to use.

### functions
```yaml
functions: list
```

List of functions with user-defined SQL functions, only used when [content](#content) is enabled. Each list element must be one of the following:

- function
- callable object
- dict with fields for name, argcount and function

[An example can be found here](../query#custom-sql-functions).

### quantize
```yaml
quantize: boolean
```

Enables quanitization of generated sentence embeddings. If the index backend supports it, sentence embeddings will be stored with 8-bit precision vs 32-bit.
Only Faiss currently supports quantization.

### tokenize
```yaml
tokenize: boolean
```

Enables string tokenization (defaults to false). This method applies tokenization rules that only work with English language text and may increase the quality of
English language sentence embeddings in some situations.

### query
```yaml
query:
    path: Sets the path for the query model. This can be any model on the
          [Hugging Face Model Hub](https://huggingface.co/models) or
          a local file path.
    prefix: text prefix to prepend to all inputs
    maxlength: maximum generated sequence length
```

Query translation model. Translates natural language queries to txtai compatible SQL statements.

## Cloud

This section describes parameters used to sync compressed indexes with cloud storage. These parameters are only enabled if an embeddings index is stored as compressed. They are set via the [embeddings.load](../methods/#txtai.embeddings.base.Embeddings.load) and [embeddings.save](../methods/#txtai.embeddings.base.Embeddings.save) methods.

### provider
```yaml
provider: string
```

The cloud storage provider, see [full list of providers here](https://libcloud.readthedocs.io/en/stable/storage/supported_providers.html).

### container
```yaml
container: string
```

Container/bucket/directory name.

### key
```yaml
key: string
```

Provider-specific access key. Can also be set via ACCESS_KEY environment variable. Ensure the configuration file is secured if added to the file.

### secret
```yaml
secret: string
```

Provider-specific access secret. Can also be set via ACCESS_SECRET environment variable. Ensure the configuration file is secured if added to the file.

### host
```yaml
host: string
```

Optional server host name. Set when using a local cloud storage server.

### port
```yaml
port: int
```

Optional server port. Set when using a local cloud storage server.

### token
```yaml
token: string
```

Optional temporary session token

### region
```yaml
region: string
```

Optional parameter to specify the storage region, provider-specific.

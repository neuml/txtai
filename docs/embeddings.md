# Embeddings
An Embeddings instance is the engine that provides similarity search. Embeddings can be used to run ad-hoc similarity comparisions or build/search large indices.

Embeddings parameters are set through the constructor. Examples below.

```python
# Transformers embeddings model
Embeddings({"method": "transformers",
            "path": "sentence-transformers/bert-base-nli-mean-tokens"})

# Word embeddings model
Embeddings({"path": vectors,
            "storevectors": True,
            "scoring": "bm25",
            "pca": 3,
            "quantize": True})
```

## Configuration

### method
```yaml
method: transformers|words
```

Sets the sentence embeddings method to use. When set to _transformers_, the embeddings object builds sentence embeddings using the sentence transformers. Otherwise a word embeddings model is used. Defaults to words.

### path
```yaml
path: string
```

Required field that sets the path for a vectors model. When method set to _transformers_, this must be a path to a Hugging Face transformers model. Otherwise,
it must be a path to a local word embeddings model.

### tokenize
```yaml
tokenize: boolean
```

Enables string tokenization (defaults to true). This method applies tokenization rules that work best with English language text and help increase the
quality of English language sentence embeddings. This should be disabled when working with non-English text.

### storevectors
```yaml
storevectors: boolean
```

Enables copying of a vectors model set in path into the embeddings models output directory on save. This option enables a fully encapsulated index with no external file dependencies.

### scoring
```yaml
scoring: bm25|tfidf|sif
```

For word embedding models, a scoring model allows building weighted averages of word vectors for a given sentence. Supports BM25, tf-idf and SIF (smooth inverse frequency) methods. If a scoring method is not provided, mean sentence embeddings are built.

### pca
```yaml
pca: int
```

Removes _n_ principal components from generated sentence embeddings. When enabled, a TruncatedSVD model is built to help with dimensionality reduction. After pooling of vectors creates a single sentence embedding, this method is applied.

### backend
```yaml
backend: annoy|faiss|hnsw
```

Approximate Nearest Neighbor (ANN) index backend for storing generated sentence embeddings. Defaults to Faiss for Linux/macOS and Annoy for Windows. Faiss currently is not supported on Windows.

Backend-specific settings are set with a corresponding configuration object having the same name as the backend (i.e. annoy, faiss, or hnsw). None of these are required and are set to defaults if omitted.

### annoy
```yaml
annoy:
    ntrees: number of trees (int) - defaults to 10
    searchk: search_k search setting (int) - defaults to -1
```

See [Annoy documentation](https://github.com/spotify/annoy#full-python-api) for more information on these parameters.

### faiss
```yaml
faiss:
    components: Comma separated list of components - defaults to None
    nprobe: search probe setting (int) - defaults to 6
```

See Faiss documentation on the [index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory) and [search](https://github.com/facebookresearch/faiss/wiki/Faster-search) for more information on these parameters.

### hnsw
```yaml
hnsw:
    efconstruction:  ef_construction param for init_index (int) - defaults to 200
    m: M param for init_index (int) - defaults to 16
    randomseed: random-seed param for init_index (init) - defaults to 100
    efsearch: ef search param (int) - defaults to None and not set
```

See [Hnswlib documentation](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) for more information on these parameters.

### quantize
```yaml
quantize: boolean
```

Enables quanitization of generated sentence embeddings. If the index backend supports it, sentence embeddings will be stored with 8-bit precision vs 32-bit.
Only Faiss currently supports quantization.

::: txtai.embeddings.Embeddings.__init__
::: txtai.embeddings.Embeddings.batchsearch
::: txtai.embeddings.Embeddings.batchsimilarity
::: txtai.embeddings.Embeddings.batchtransform
::: txtai.embeddings.Embeddings.index
::: txtai.embeddings.Embeddings.load
::: txtai.embeddings.Embeddings.save
::: txtai.embeddings.Embeddings.score
::: txtai.embeddings.Embeddings.search
::: txtai.embeddings.Embeddings.similarity
::: txtai.embeddings.Embeddings.transform

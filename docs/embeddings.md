# Embeddings
Embeddings is the engine that delivers semantic search. Text is transformed into embeddings vectors where similar concepts will produce similar vectors. Indexes both large and small are built with these vectors. The indexes are used find results that have the same meaning, not necessarily the same keywords.

Embeddings parameters are set through the constructor. Examples below.

```python
# Transformers embeddings model
Embeddings({"method": "transformers",
            "path": "sentence-transformers/nli-mpnet-base-v2"})

# Word embeddings model
Embeddings({"method": "words",
            "path": vectors,
            "storevectors": True,
            "scoring": "bm25",
            "pca": 3,
            "quantize": True})
```

## Configuration

### method
```yaml
method: transformers|sentence-transformers|words
```

Sentence embeddings method to use. Options listed below.

#### transformers

Builds sentence embeddings using a transformers model. While this can be any transformers model, it works best with
[models trained](https://huggingface.co/models?pipeline_tag=sentence-similarity) to build sentence embeddings.

#### sentence-transformers

Same as transformers but loads models with the sentence-transformers library.

#### words

Builds sentence embeddings using a word embeddings model.

sentence-transformers and words require the [similarity](https://neuml.github.io/txtai/install/#similarity) extras package to be installed.

The method is inferred using the _path_ if not provided.

### path
```yaml
path: string
```

Required field that sets the path for a vectors model. When using a transformers/sentence-transformers model, this can be any model on the
[Hugging Face Model Hub](https://huggingface.co/models) or a local file path. Otherwise, it must be a local file path to a word embeddings model.

### backend
```yaml
backend: faiss|hnsw|annoy
```

Approximate Nearest Neighbor (ANN) index backend for storing generated sentence embeddings. Defaults to Faiss. Additional backends require the
[similarity](https://neuml.github.io/txtai/install/#similarity) extras package to be installed.

Backend-specific settings are set with a corresponding configuration object having the same name as the backend (i.e. annoy, faiss, or hnsw). None of these are required and are set to defaults if omitted.

### faiss
```yaml
faiss:
    components: Comma separated list of components - defaults to "Flat" for small indices and "IVFx,Flat" for larger indexes where x = 4 * sqrt(embeddings count)
    nprobe: search probe setting (int) - defaults to x/16 (as defined above) for larger indexes
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
    randomseed: random-seed param for init_index (init) - defaults to 100
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

### quantize
```yaml
quantize: boolean
```

Enables quanitization of generated sentence embeddings. If the index backend supports it, sentence embeddings will be stored with 8-bit precision vs 32-bit.
Only Faiss currently supports quantization.

## Additional configuration for Transformers models

### tokenize
```yaml
tokenize: boolean
```

Enables string tokenization (defaults to false). This method applies tokenization rules that only work with English language text and may increase the quality of
English language sentence embeddings in some situations.

## Additional configuration for Word embedding models

Word embeddings provide a good tradeoff of performance to functionality for a similarity search system. With that being said, Transformers models are making great progress in scaling performance down to smaller models and are the preferred vector backend in txtai for most cases.

Word embeddings models require the [similarity](https://neuml.github.io/txtai/install/#similarity) extras package to be installed.

### storevectors
```yaml
storevectors: boolean
```

Enables copying of a vectors model set in path into the embeddings models output directory on save. This option enables a fully encapsulated index with no external file dependencies.

### scoring
```yaml
scoring: bm25|tfidf|sif
```

A scoring model builds weighted averages of word vectors for a given sentence. Supports BM25, TF-IDF and SIF (smooth inverse frequency) methods. If a scoring method is not provided, mean sentence embeddings are built.

### pca
```yaml
pca: int
```

Removes _n_ principal components from generated sentence embeddings. When enabled, a TruncatedSVD model is built to help with dimensionality reduction. After pooling of vectors creates a single sentence embedding, this method is applied.

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

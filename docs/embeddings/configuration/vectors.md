# Vectors

The following covers available vector model configuration options.

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

`sentence-transformers` and `words` require the [similarity](../../../install/#similarity) extras package to be installed.

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

#### gpu
```yaml
gpu: boolean|int|string|device
```

Set the target device. Supports true/false, device id, device string and torch device instance. This is automatically derived if omitted.

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

## quantize
```yaml
quantize: int|bool
```

Enables scalar quantization at the specified precision. Supports 1-bit through 8-bit quantization. Scalar quantization transforms continuous floating point values
to discrete unsigned integers. Quantized data storage is only supported by the `faiss`, `numpy` and `torch` ANN backends.

This parameter supports booleans for backwards compatability. When set to true/false, this flag sets [faiss.quantize](../ann/#faiss).

## instructions
```yaml
instructions:
    query: prefix for queries
    data: prefix for indexing
```

Instruction-based models use prefixes to modify how embeddings are computed. This is especially useful with asymmetric search, which is when the query and indexed data are of vastly different lengths. In other words, short queries with long documents.

[E5-base](https://huggingface.co/intfloat/e5-base) is an example of a model that accepts instructions. It takes `query: ` and `passage: ` prefixes and uses those to generate embeddings that work well for asymmetric search.

## models
```yaml
models: dict
```

Loads and stores vector models in this cache. This is primarily used with subindexes but can be set on any embeddings instance. This prevents the same model from being loaded multiple times when working with multiple embeddings instances.

## tokenize
```yaml
tokenize: boolean
```

Enables string tokenization (defaults to false). This method applies tokenization rules that only work with English language text. It's not recommended for use with recent vector models.

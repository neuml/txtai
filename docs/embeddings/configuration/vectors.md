# Vectors

The following covers available vector model configuration options.

## path
```yaml
path: string
```

Sets the path for a vectors model. When using a transformers/sentence-transformers model, this can be any model on the
[Hugging Face Hub](https://huggingface.co/models) or a local file path. Otherwise, it must be a local file path to a word embeddings model.

## defaults
```yaml
defaults: boolean
```

Uses default vector model path when enabled (default setting is True) and `path` is not provided. See [this link](../) for an example.

## method
```yaml
method: transformers|sentence-transformers|llama.cpp|litellm|model2vec|external|words
```

Embeddings method to use. If the method is not provided, it is inferred using the `path`.

`sentence-transformers`, `llama.cpp`, `litellm`, `model2vec` and `words` require the [vectors](../../../install/#vectors) extras package to be installed.

### transformers

Builds embeddings using a transformers model. While this can be any transformers model, it works best with
[models trained](https://huggingface.co/models?pipeline_tag=sentence-similarity) to build embeddings.

Both `mean` and `cls` pooling are supported and automatically inferred from the model. The pooling method can be overwritten by changing the method
from `transformers` to `meanpooling` or `clspooling` respectively.

Setting `maxlength` to `True` enables truncating inputs to the `max_seq_length`. Setting `maxlength` to an integer will truncate inputs to that value. When omitted (default), the `maxlength` will be set to either the model or tokenizer maxlength.

### sentence-transformers

Same as transformers but loads models with the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library.

### llama.cpp

Builds embeddings using a [llama.cpp](https://github.com/abetlen/llama-cpp-python) model. Supports both local and remote GGUF paths on the HF Hub.

### litellm

Builds embeddings using a LiteLLM model. See the [LiteLLM documentation](https://litellm.vercel.app/docs/providers) for the options available with LiteLLM models.

### model2vec

Builds embeddings using a [Model2Vec](https://github.com/MinishLab/model2vec) model. Model2Vec is a knowledge-distilled version of a transformers model with static vectors.

### words

Builds embeddings using a word embeddings model and static vectors. While Transformers models are preferred in most cases, this method can be useful for low resource and historical languages where there isn't much linguistic data available.

#### pca
```yaml
pca: int
```

Removes _n_ principal components from generated embeddings. When enabled, a TruncatedSVD model is built to help with dimensionality reduction. After pooling of vectors creates a single embedding, this method is applied.

### external

Embeddings are created via an external model or API. Requires setting the [transform](#transform) parameter to a function that translates data into embeddings.

#### transform
```yaml
transform: function
```

When method is `external`, this function transforms input content into embeddings. The input to this function is a list of data. This method must return either a numpy array or list of numpy arrays.

## gpu
```yaml
gpu: boolean|int|string|device
```

Set the target device. Supports true/false, device id, device string and torch device instance. This is automatically derived if omitted.

The `sentence-transformers` method supports encoding with multiple GPUs. This can be enabled by setting the gpu parameter to `all`.

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

## dimensionality
```yaml
dimensionality: int
```

Enables truncation of vectors to this dimensionality. This is only useful for models trained to store more important information in earlier dimensions such as [Matryoshka Representation Learning (MRL)](https://huggingface.co/blog/matryoshka).

## quantize
```yaml
quantize: int|boolean
```

Enables scalar vector quantization at the specified precision. Supports 1-bit through 8-bit quantization. Scalar quantization transforms continuous floating point values to discrete unsigned integers. The `faiss`, `pgvector`, `numpy` and `torch` ANN backends support storing these vectors.

This parameter supports booleans for backwards compatability. When set to true/false, this flag sets [faiss.quantize](../ann/#faiss).

In addition to vector-level quantization, some ANN backends have the ability to quantize vectors at the storage layer. See the [ANN](../ann) configuration options for more.

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

## vectors
```yaml
vectors: dict
```

Passes these additional parameters to the underlying vector model.

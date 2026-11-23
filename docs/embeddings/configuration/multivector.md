# Multi-vector

Enable stored token vectors via the `multivector` parameter. `Embeddings` owns this optional component alongside ANN, scoring, indexes and graph. Search does not call it.

A dictionary or `true` enables the component. `false`, `null` and `0` disable it. `true` and `{}` use `LateEncoder` defaults.

## path
```yaml
path: model path
```

Sets the late interaction model path. Defaults to `colbert-ir/colbertv2.0`, independently of the primary embeddings model.

## gpu
```yaml
gpu: boolean|int|string|device
```

Sets the encoder device. Defaults to automatic device selection.

## method
```yaml
method: model pooling method
```

Sets the pooling implementation passed to `LateEncoder`. When omitted, the model configuration determines it.

## tokenizer
```yaml
tokenizer: tokenizer options
```

Sets tokenizer options passed to the encoder.

## maxlength
```yaml
maxlength: int
```

Sets the encoder token length limit.

## vectors
```yaml
vectors: additional model arguments
```

Sets model arguments passed to the late encoder. `muvera` and `lemur` are disabled by `LateEncoder` to retain token rows.

## Usage
```python
from txtai import Embeddings

embeddings = Embeddings(
    content=True,
    multivector={"path": "neuml/colbert-bert-tiny", "gpu": False},
)
embeddings.index(["lottery ticket", "ice shelf"])
results = embeddings.search("lottery ticket", 20)
ids = [row["id"] for row in results]
scores = embeddings.multivector.score("lottery ticket", ids, limit=5)
```

With content disabled, use `ids = [uid for uid, _ in results]`. `score` returns `(id, score)` pairs in descending order. Unknown, deleted and non-text ids are omitted. Scoring delegates to `LateEncoder.score`, using mean MaxSim over true query tokens.

## Storage

The component stores per-row symmetric int8 token values, one fp16 scale per row and int64 offsets marking where each document's token rows start and end. It saves one `multivector` safetensors file containing `tokens` (int8), `scales` (float16) and `offsets` (int64).

The component follows index, upsert, delete, reindex, save, load and close. Archive and cloud saves include the file. Loading reads the arrays in full and creates the encoder. Indexing also encodes text with the late interaction model. Non-text objects occupy empty token slots, and lists of text tokens are joined with spaces.

When top-level indexing is enabled, a document with neither text nor an object consumes no vector slot. When top-level indexing is disabled, `Transform` substitutes the document id as text and that text receives token rows.

When the key is absent or disabled, no component file is saved. The component loads its own encoder and scans the id mapping when content is disabled.

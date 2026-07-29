# LemurTrainer

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

Trains a LEMUR fixed dimensional encoder for a late interaction model and corpus. Training is a separate pipeline step; the saved
artifact is then loaded through the embeddings `vectors.lemur` configuration.

## Example

```python
from txtai.pipeline import LemurTrainer

corpus = [
    "First document",
    "Second document",
    "Third document",
]

trainer = LemurTrainer()
trainer(
    "colbert-ir/colbertv2.0",
    corpus,
    "lemur-model",
    epochs=100,
    validation_split=0.1,
)
```

`epochs` is required so the training mode is explicit. Use `epochs=100` for the quality-oriented MLP setting. This can take hours on
a CPU. Use `epochs=0` to select deterministic random ELM features as a lower-cost fallback.

The learn distribution defaults to `learn_category="query"`. Target documents are always encoded with the data encoder, so the
default encodes selected corpus texts once as data and once as queries. Set `learn_category="data"` to reuse the data encodings, or
pass a separate iterable of texts with `learn`.

Set `corpus_subset_size` to a positive integer to sample that many raw corpus texts under `seed` before either encoding pass. The
default is `None`, which uses every corpus text. `train_subset_size` and `learn_subset_size` apply later, after token vectors have
already been created. `corpus_subset_size` applies to `data`; a separate `learn` iterable remains caller-sized.

For trained MLP features, set `validation_split` to a fraction greater than zero and less than one to retain the epoch with the lowest
held-out loss. The default is `0.0`, which preserves training-loss selection. The selected one-based epoch, loss and metric are
available as `selected_epoch`, `selected_loss` and `selection_metric` on the fitted or reloaded encoder.

The artifact contains `config.json` and `model.safetensors`. It stores the inference feature model, output-normalization statistics
and token sample needed to encode documents added after training. The training-only output readout is not saved.

Load the artifact in an embeddings configuration.

```yaml
embeddings:
    path: colbert-ir/colbertv2.0
    vectors:
        lemur:
            path: lemur-model
```

## Search behavior

LEMUR approximates standardized MaxSim targets, so useful ranking scores can be negative. txtai's dense vector path L2-normalizes the
fixed vectors and removes results with scores less than or equal to zero. This can change ordering and return fewer than the requested
number of candidates compared with raw maximum inner-product search.

The Faiss backend uses exact `IDMap,Flat` search through 5,000 rows and switches to an IVF index above that threshold. In the measured
scifact run, default IVF reduced LEMUR NDCG@10 by 43% relative to exact search, compared with 25% for MUVERA. For LEMUR corpora above
5,000 rows, either pin `faiss.components` to an exact index or tune the IVF settings for the corpus. The exact configuration is:

```yaml
embeddings:
    path: colbert-ir/colbertv2.0
    vectors:
        lemur:
            path: lemur-model
    faiss:
        components: IDMap,Flat
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.LemurTrainer.__call__

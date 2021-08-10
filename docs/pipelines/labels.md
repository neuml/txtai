# Labels

A Labels pipeline uses a text classification model to apply labels to input text. This pipeline can classify text using either a zero shot model (dynamic labeling) or a standard text classification model (fixed labeling).

Labels parameters are set as constructor arguments. Examples below.

```python
# Default configuration
Labels()

# Custom model with zero shot classification
Labels("roberta-large-mnli")

# Model with fixed labels
Labels(dynamic=False)
```

::: txtai.pipeline.Labels.__init__
::: txtai.pipeline.Labels.__call__

# Extractor

An Extractor pipeline is a combination of an embeddings query and an Extractive QA model. Filtering the context for a QA model helps maximize performance of the model.

Extractor parameters are set as constructor arguments. Examples below.

```python
Extractor(embeddings, path, quantize, gpu, model, tokenizer)
```

::: txtai.pipeline.Extractor.__init__
::: txtai.pipeline.Extractor.__call__

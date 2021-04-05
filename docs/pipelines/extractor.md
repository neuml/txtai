# Extractor

An Extractor pipeline is a combination of an embeddings query and an Extractive QA model. Filtering the context for a QA model helps maximize performance of the model.

Extractor parameters are set as constructor arguments. Examples below.

```python
Extractor(embeddings, path, quantize, gpu, model, tokenizer)
```

## embeddings
```yaml
embeddings: Embeddings object instance
```

Embeddings object instance. Used to query and find candidate text snippets to run the question-answer model against.

## tokenizer
```yaml
tokenizer: Tokenizer function
```

Optional custom tokenizer function to parse input queries

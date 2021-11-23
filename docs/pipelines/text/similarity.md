## Similarity

A Similarity pipeline is also a zero shot classifier model where the labels are the queries. The results are transposed to get scores per query/label vs scores per input text. 

Similarity parameters are set as constructor arguments. Examples below.

```python
Similarity()
Similarity("roberta-large-mnli")
```

::: txtai.pipeline.Similarity.__init__
::: txtai.pipeline.Similarity.__call__

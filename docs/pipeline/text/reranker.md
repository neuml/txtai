# Reranker

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Reranker pipeline runs embeddings queries and re-ranks them using a similarity pipeline. 

## Example

The following shows a simple example using this pipeline.

```python
from txtai import Embeddings
from txtai.pipeline import Reranker, Similarity

# Embeddings instance
embeddings = Embeddings()
embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")

# Similarity instance
similarity = Similarity(path="colbert-ir/colbertv2.0", lateencode=True)

# Reranking pipeline
reranker = Reranker(embeddings, similarity)
reranker("Tell me about AI")
```

_Note: Content must be enabled with the embeddings instance for this to work properly._

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
embeddings:

similarity:

# Create pipeline using lower case class name
reranker:

# Run pipeline with workflow
workflow:
  translate:
    tasks:
      - reranker
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("reranker", ["Tell me about AI"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"rerank", "elements":["Tell me about AI"]}'
```

## Methods 

Python documentation for the pipeline.

### ::: txtai.pipeline.Reranker.__init__
### ::: txtai.pipeline.Reranker.__call__

# Similarity

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Similarity pipeline computes similarity between queries and list of text using a text classifier.

This pipeline supports both standard text classification models and zero-shot classification models. The pipeline uses the queries as labels for the input text. The results are transposed to get scores per query/label vs scores per input text. 

Cross-encoder models are supported via the `crossencode=True` constructor parameter. These models are loaded with a CrossEncoder pipeline that can also be instantiated directly. The CrossEncoder pipeline has the same methods and functionality as described below.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Similarity

# Create and run pipeline
similarity = Similarity()
similarity("feel good story", [
    "Maine man wins $1M from $25 lottery ticket", 
    "Don't sacrifice slower friends in a bear attack"
])
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Add semantic search to Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb)  | Add semantic search to existing search systems | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
similarity:
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
app.similarity("feel good story", [
    "Maine man wins $1M from $25 lottery ticket", 
    "Don't sacrifice slower friends in a bear attack"
])
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/similarity" \
  -H "Content-Type: application/json" \
  -d '{"query": "feel good story", "texts": ["Maine man wins $1M from $25 lottery ticket", "Dont sacrifice slower friends in a bear attack"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Similarity.__init__
### ::: txtai.pipeline.Similarity.__call__

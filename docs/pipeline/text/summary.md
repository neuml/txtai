# Summary

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Summary pipeline summarizes text. This pipeline runs a text2text model that abstractively creates a summary of the input text.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Summary

# Create and run pipeline
summary = Summary()
summary("Enter long, detailed text to summarize here")
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Building abstractive text summaries](https://github.com/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) | Run abstractive text summarization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
summary:

# Run pipeline with workflow
workflow:
  summary:
    tasks:
      - action: summary
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("summary", ["Enter long, detailed text to summarize here"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"summary", "elements":["Enter long, detailed text to summarize here"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Summary.__init__
### ::: txtai.pipeline.Summary.__call__

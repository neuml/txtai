# Labels

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Labels pipeline uses a text classification model to apply labels to input text. This pipeline can classify text using either a zero shot model (dynamic labeling) or a standard text classification model (fixed labeling).

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Labels

# Create and run pipeline
labels = Labels()
labels(
    ["Great news", "That's rough"],
    ["positive", "negative"]
)
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Apply labels with zero shot classification](https://github.com/neuml/txtai/blob/master/examples/07_Apply_labels_with_zero_shot_classification.ipynb) | Use zero shot learning for labeling, classification and topic modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/07_Apply_labels_with_zero_shot_classification.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
labels:

# Run pipeline with workflow
workflow:
  labels:
    tasks:
      - action: labels
        args: [["positive", "negative"]]
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("labels", ["Great news", "That's rough"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"labels", "elements": ["Great news", "Thats rough"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Labels.__init__
### ::: txtai.pipeline.Labels.__call__

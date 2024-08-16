# Objects

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Objects pipeline reads a list of images and returns a list of detected objects.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Objects

# Create and run pipeline
objects = Objects()
objects("path to image file")
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Generate image captions and detect objects](https://github.com/neuml/txtai/blob/master/examples/25_Generate_image_captions_and_detect_objects.ipynb) | Captions and object detection for images | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/25_Generate_image_captions_and_detect_objects.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
objects:

# Run pipeline with workflow
workflow:
  objects:
    tasks:
      - action: objects
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("objects", ["path to image file"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"objects", "elements":["path to image file"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Objects.__init__
### ::: txtai.pipeline.Objects.__call__

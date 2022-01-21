# Caption

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The caption pipeline reads a list of images and returns a list of captions for those images.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Caption

# Create and run pipeline
caption = Caption()
caption("path to image file")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api).

### config.yml
```yaml
# Create pipeline using lower case class name
caption:

# Run pipeline with workflow
workflow:
  caption:
    tasks:
      - action: caption
```

### Run with Workflows

```python
from txtai.api import API

# Create and run pipeline with workflow
app = API("config.yml")
list(app.workflow("caption", ["path to image file"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"caption", "elements":["path to image file"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Caption.__init__
### ::: txtai.pipeline.Caption.__call__

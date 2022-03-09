# ImageHash

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The image hash pipeline generates perceptual image hashes. These hashes can be used to detect near-duplicate images. This method is not backed by machine learning models and not intended to find conceptually similar images.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import ImageHash

# Create and run pipeline
ihash = ImageHash()
ihash("path to image file")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
imagehash:

# Run pipeline with workflow
workflow:
  imagehash:
    tasks:
      - action: imagehash
```

### Run with Workflows

```python
from txtai.app import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("imagehash", ["path to image file"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"imagehash", "elements":["path to image file"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.ImageHash.__init__
### ::: txtai.pipeline.ImageHash.__call__

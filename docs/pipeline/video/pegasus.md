# Pegasus

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Pegasus pipeline analyzes videos and generates text using the [TwelveLabs](https://twelvelabs.io) Pegasus video understanding model via the TwelveLabs API. Given a video and a prompt, Pegasus can summarize, caption, answer questions about or extract structured information from the video.

A video is referenced by a public url or, for previously uploaded content, a TwelveLabs asset id passed as `{"asset_id": "..."}`.

This pipeline requires a TwelveLabs API key, read from the `api_key` parameter or the `TWELVELABS_API_KEY` environment variable. A free API key with a generous free tier is available at [twelvelabs.io](https://twelvelabs.io).

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Pegasus

# Create and run pipeline
pegasus = Pegasus()
pegasus("https://example.com/sample.mp4", "Describe what happens in this video")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
pegasus:

# Run pipeline with workflow
workflow:
  pegasus:
    tasks:
      - action: pegasus
        args: ["Describe what happens in this video"]
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("pegasus", ["https://example.com/sample.mp4"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"pegasus", "elements":["https://example.com/sample.mp4"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Pegasus.__init__
### ::: txtai.pipeline.Pegasus.__call__

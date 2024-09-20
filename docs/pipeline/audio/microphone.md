# Microphone

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Microphone pipeline reads input audio from a microphone device. This pipeline is designed to run on local machines given that it requires access to read from an input device.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Microphone

# Create and run pipeline
microphone = Microphone()
microphone()
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
audiostream:

# Run pipeline with workflow
workflow:
  audiostream:
    tasks:
      - action: audiostream
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("audiostream", ["numpy data"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"audiostream", "elements":["numpy data"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Microphone.__init__
### ::: txtai.pipeline.Microphone.__call__

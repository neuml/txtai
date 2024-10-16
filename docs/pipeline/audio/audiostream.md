# Audio Stream

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Audio Stream pipeline is a threaded pipeline that plays audio segments. This pipeline is designed to run on local machines given that it requires access to write to an output device.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import AudioStream

# Create and run pipeline
audio = AudioStream()
audio(data)
```

This pipeline may require additional system dependencies. See [this section](../../../install#environment-specific-prerequisites) for more.

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Speech to Speech RAG](https://github.com/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) [▶️](https://www.youtube.com/watch?v=tH8QWwkVMKA) | Full cycle speech to speech workflow with RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) |

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
list(app.workflow("audiostream", [["numpy data", "sample rate"]]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"audiostream", "elements":[["numpy data", "sample rate"]]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.AudioStream.__init__
### ::: txtai.pipeline.AudioStream.__call__

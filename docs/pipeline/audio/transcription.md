# Transcription

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Transcription pipeline converts speech in audio files to text.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Transcription

# Create and run pipeline
transcribe = Transcription()
transcribe("path to wav file")
```

This pipeline may require additional system dependencies. See [this section](../../../install#environment-specific-prerequisites) for more.

See the links below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Transcribe audio to text](https://github.com/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) | Convert audio files to text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) |
| [Speech to Speech RAG](https://github.com/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) [▶️](https://www.youtube.com/watch?v=tH8QWwkVMKA) | Full cycle speech to speech workflow with RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
transcription:

# Run pipeline with workflow
workflow:
  transcribe:
    tasks:
      - action: transcription
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("transcribe", ["path to wav file"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"transcribe", "elements":["path to wav file"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Transcription.__init__
### ::: txtai.pipeline.Transcription.__call__

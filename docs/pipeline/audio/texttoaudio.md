# Text To Audio

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Text To Audio pipeline generates audio from text.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import TextToAudio

# Create and run pipeline
tta = TextToAudio()
tta("Describe the audio to generate here")
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Generative Audio](https://github.com/neuml/txtai/blob/master/examples/66_Generative_Audio.ipynb) | Storytelling with generative audio workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/66_Generative_Audio.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
texttoaudio:

# Run pipeline with workflow
workflow:
  tta:
    tasks:
      - action: texttoaudio
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("tta", ["Describe the audio to generate here"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"tta", "elements":["Describe the audio to generate here"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.TextToAudio.__init__
### ::: txtai.pipeline.TextToAudio.__call__

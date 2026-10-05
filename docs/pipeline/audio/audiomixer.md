# Audio Mixer

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Audio Mixer pipeline mixes multiple audio streams into a single stream.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import AudioMixer

# Create and run pipeline
mixer = AudioMixer()
mixer(((audio1, rate1), (audio2, rate2)))
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
audiomixer:

# Run pipeline with workflow
workflow:
  audiomixer:
    tasks:
      - action: audiomixer
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("audiomixer", [[[audio1, rate1], [audio2, rate2]]]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"audiomixer", "elements":[[[audio1, rate1], [audio2, rate2]]]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.AudioMixer.__init__
### ::: txtai.pipeline.AudioMixer.__call__

# Text To Speech

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Text To Speech pipeline generates speech from text.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import TextToSpeech

# Create and run pipeline with default model
tts = TextToSpeech()
tts("Say something here")

# Stream audio - incrementally generates snippets of audio
yield from tts(
  "Say something here. And say something else.".split(),
  stream=True
)

# Generate audio using a speaker id
tts = TextToSpeech("neuml/vctk-vits-onnx")
tts("Say something here", speaker=15)

# Generate audio using speaker embeddings
tts = TextToSpeech("neuml/txtai-speecht5-onnx")
tts("Say something here", speaker=np.array(...))
```

See the links below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Text to speech generation](https://github.com/neuml/txtai/blob/master/examples/40_Text_to_Speech_Generation.ipynb) | Generate speech from text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/40_Text_to_Speech_Generation.ipynb) |
| [Speech to Speech RAG](https://github.com/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) [▶️](https://www.youtube.com/watch?v=tH8QWwkVMKA) | Full cycle speech to speech workflow with RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) |
| [Generative Audio](https://github.com/neuml/txtai/blob/master/examples/66_Generative_Audio.ipynb) | Storytelling with generative audio workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/66_Generative_Audio.ipynb) |

This pipeline is backed by ONNX models from the Hugging Face Hub. The following models are currently available.

- [kokoro-base-onnx](https://huggingface.co/NeuML/kokoro-base-onnx) | [fp16](https://huggingface.co/NeuML/kokoro-fp16-onnx) | [int8](https://huggingface.co/NeuML/kokoro-int8-onnx)
- [ljspeech-jets-onnx](https://huggingface.co/NeuML/ljspeech-jets-onnx)
- [ljspeech-vits-onnx](https://huggingface.co/NeuML/ljspeech-vits-onnx)
- [vctk-vits-onnx](https://huggingface.co/NeuML/vctk-vits-onnx)
- [txtai-speecht5-onnx](https://huggingface.co/NeuML/txtai-speecht5-onnx)

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
texttospeech:

# Run pipeline with workflow
workflow:
  tts:
    tasks:
      - action: texttospeech
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("tts", ["Say something here"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"tts", "elements":["Say something here"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.TextToSpeech.__init__
### ::: txtai.pipeline.TextToSpeech.__call__

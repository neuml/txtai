# Translation

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Translation pipeline translates text between languages. It supports over 100+ languages. Automatic source language detection is built-in. This pipeline detects the language of each input text row, loads a model for the source-target combination and translates text to the target language.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Translation

# Create and run pipeline
translate = Translation()
translate("This is a test translation into Spanish", "es")
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Translate text between languages](https://github.com/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) | Streamline machine translation and language detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
translation:

# Run pipeline with workflow
workflow:
  translate:
    tasks:
      - action: translation
        args: ["es"]
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("translate", ["This is a test translation into Spanish"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"translate", "elements":["This is a test translation into Spanish"]}'
```

## Methods 

Python documentation for the pipeline.

### ::: txtai.pipeline.Translation.__init__
### ::: txtai.pipeline.Translation.__call__

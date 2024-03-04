# Textractor

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Textractor pipeline extracts and splits text from documents. This pipeline uses either an [Apache Tika](https://github.com/chrismattmann/tika-python) backend (if Java is available) or [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/).

_Note: BeautifulSoup4 only supports HTML documents, anything else requires Tika and Java to be installed._

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Textractor

# Create and run pipeline
textract = Textractor()
textract("https://github.com/neuml/txtai")
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Extract text from documents](https://github.com/neuml/txtai/blob/master/examples/10_Extract_text_from_documents.ipynb) | Extract text from PDF, Office, HTML and more | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/10_Extract_text_from_documents.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
textractor:

# Run pipeline with workflow
workflow:
  textract:
    tasks:
      - action: textractor
```

### Run with Workflows

```python
from txtai.app import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("textract", ["https://github.com/neuml/txtai"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"textract", "elements":["https://github.com/neuml/txtai"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Textractor.__init__
### ::: txtai.pipeline.Textractor.__call__

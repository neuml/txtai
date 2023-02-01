# Extractor

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Extractor pipeline is a combination of a similarity instance (embeddings or similarity pipeline) to build a question context and a model that answers questions. The model can be a prompt-driven large language model (LLM), an extractive question-answering model or a custom pipeline.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.embeddings import Embeddings
from txtai.pipeline import Extractor

# Embeddings model ranks candidates before passing to QA pipeline
embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})

# Create and run pipeline
extractor = Extractor(embeddings, "distilbert-base-cased-distilled-squad")
extractor([["What was won"] * 3 + [False]],
          ["Maine man wins $1M from $25 lottery ticket"])
```

See the links below for more detailed examples.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Extractive QA with txtai](https://github.com/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) | Introduction to extractive question-answering with txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) |
| [Extractive QA with Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) | Run extractive question-answering queries with Elasticsearch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) |
| [Extractive QA to build structured data](https://github.com/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) | Build structured datasets using extractive question-answering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) |
| [Prompt-driven search with LLMs](https://github.com/neuml/txtai/blob/master/examples/42_Prompt_driven_search_with_LLMs.ipynb) | Embeddings-guided and Prompt-driven search with Large Language Models (LLMs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/42_Prompt_driven_search_with_LLMs.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
extractor:
```

### Run with Workflows

```python
from txtai.app import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.extract([{"name": "What was won", "query": "What was won",
                   "question", "What was won", "snippet": False}], 
                 ["Maine man wins $1M from $25 lottery ticket"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"queue": [{"name":"What was won", "query": "What was won", "question": "What was won", "snippet": false}], "texts": ["Maine man wins $1M from $25 lottery ticket"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Extractor.__init__
### ::: txtai.pipeline.Extractor.__call__

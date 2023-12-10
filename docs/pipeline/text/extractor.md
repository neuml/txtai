# Extractor

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Extractor pipeline joins a prompt, context data store and generative model together to extract knowledge.

The data store can be an embeddings database or a similarity instance with associated input text. The generative model can be a prompt-driven large language model (LLM), an extractive question-answering model or a custom pipeline. This is known as prompt-driven search or retrieval augmented generation (RAG).

## Example

The following shows a simple example using this pipeline.

```python
from txtai.embeddings import Embeddings
from txtai.pipeline import Extractor

# LLM prompt
def prompt(question):
  return f"""
  Answer the following question using the provided context.

  Question:
  {question}

  Context:
  """

# Input data
data = [
  "US tops 5 million confirmed virus cases",
  "Canada's last fully intact ice shelf has suddenly collapsed, " +
  "forming a Manhattan-sized iceberg",
  "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
  "The National Park Service warns against sacrificing slower friends " +
  "in a bear attack",
  "Maine man wins $1M from $25 lottery ticket",
  "Make huge profits without work, earn up to $100,000 a day"
]

# Build embeddings index
embeddings = Embeddings({"content": True})
embeddings.index([(uid, text, None) for uid, text in enumerate(data)])

# Create and run pipeline
extractor = Extractor(embeddings, "google/flan-t5-base")
extractor([{"query": "What was won?", "question": prompt("What was won?")}])
```

See the links below for more detailed examples.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Prompt-driven search with LLMs](https://github.com/neuml/txtai/blob/master/examples/42_Prompt_driven_search_with_LLMs.ipynb) | Embeddings-guided and Prompt-driven search with Large Language Models (LLMs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/42_Prompt_driven_search_with_LLMs.ipynb) |
| [Prompt templates and task chains](https://github.com/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) | Build model prompts and connect tasks together with workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) |
| [Build RAG pipelines with txtai](https://github.com/neuml/txtai/blob/master/examples/52_Build_RAG_pipelines_with_txtai.ipynb) | Guide on retrieval augmented generation including how to create citations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/52_Build_RAG_pipelines_with_txtai.ipynb) |
| [Integrate LLM frameworks](https://github.com/neuml/txtai/blob/master/examples/53_Integrate_LLM_Frameworks.ipynb) | Integrate llama.cpp, LiteLLM and custom generation frameworks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/53_Integrate_LLM_Frameworks.ipynb) |
| [Extractive QA with txtai](https://github.com/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) | Introduction to extractive question-answering with txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) |
| [Extractive QA with Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) | Run extractive question-answering queries with Elasticsearch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) |
| [Extractive QA to build structured data](https://github.com/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) | Build structured datasets using extractive question-answering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Allow documents to be indexed
writable: True

# Content is required for extractor pipeline
embeddings:
  content: True

extractor:
  path: google/flan-t5-base

workflow:
  search:
    tasks:
      - task: extractor
        template: |
          Answer the following question using the provided context.

          Question:
          {text}

          Context:
        action: extractor
```

### Run with Workflows

Built in tasks make using the extractor pipeline easier.

```python
from txtai.app import Application

# Create and run pipeline with workflow
app = Application("config.yml")
app.add([
  "US tops 5 million confirmed virus cases",
  "Canada's last fully intact ice shelf has suddenly collapsed, " +
  "forming a Manhattan-sized iceberg",
  "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
  "The National Park Service warns against sacrificing slower friends " +
  "in a bear attack",
  "Maine man wins $1M from $25 lottery ticket",
  "Make huge profits without work, earn up to $100,000 a day"
])
app.index()

list(app.workflow("search", ["What was won?"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name": "search", "elements": ["What was won"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Extractor.__init__
### ::: txtai.pipeline.Extractor.__call__

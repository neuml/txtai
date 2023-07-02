# LLM

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The LLM pipeline runs prompts through a large language model (LLM). This pipeline autodetects if the model path is a text generation or sequence to sequence model. 

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import LLM

# Create and run LLM pipeline
llm = LLM()
llm(
  """
  Answer the following question using the provided context.

  Question:
  What are the applications of txtai?

  Context:
  txtai is an open-source platform for semantic search and
  workflows powered by language models.
  """
)
```

The LLM pipeline automatically detects the underlying model type (`text-generation` or `sequence-sequence`). This can also be manually set.

```python
from txtai.pipeline import LLM, Generator, Sequences

# Set model type via task parameter
llm = LLM("google/flan-t5-xl", task="sequence-sequence")

# Create sequences pipeline (same as previous statement)
sequences = Sequences("google/flan-t5-xl")

# Set model type via task parameter
llm = LLM("openlm-research/open_llama_3b", task="language-generation")

# Create generator pipeline (same as previous statement)
generator = Generator("openlm-research/open_llama_3b")
```

As with other pipelines, models can be externally loaded and passed to pipelines. This is useful for models that are not yet supported by Transformers.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

# Load Falcon-7B-Instruct
path = "tiiuae/falcon-7b-instruct"
model = AutoModelForCausalLM.from_pretrained(
  path,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(path)

llm = LLM((model, tokenizer))
```

See the links below for more detailed examples.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Prompt-driven search with LLMs](https://github.com/neuml/txtai/blob/master/examples/42_Prompt_driven_search_with_LLMs.ipynb) | Embeddings-guided and Prompt-driven search with Large Language Models (LLMs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/42_Prompt_driven_search_with_LLMs.ipynb) |
| [Prompt templates and task chains](https://github.com/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) | Build model prompts and connect tasks together with workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name. Use `generator` or `sequences` to force model type.
llm:

# Run pipeline with workflow
workflow:
  llm:
    tasks:
      - action: llm
```

### Run with Workflows

```python
from txtai.app import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("llm", [
  """
  Answer the following question using the provided context.
 
  Question:
  What are the applications of txtai? 

  Context:
  txtai is an open-source platform for semantic search and
  workflows powered by language models.
  """
]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"sequences", "elements": ["Answer the following question..."]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.LLM.__init__
### ::: txtai.pipeline.LLM.__call__

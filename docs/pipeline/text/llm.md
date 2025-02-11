# LLM

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The LLM pipeline runs prompts through a large language model (LLM). This pipeline autodetects the LLM framework based on the model path.

## Example

The following shows a simple example using this pipeline.

```python
from txtai import LLM

# Create LLM pipeline
llm = LLM()

# Run prompt
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

# Instruction tuned models typically require string prompts to
# follow a specific chat template set by the model
llm(
  """
  <|im_start|>system
  You are a friendly assistant.<|im_end|>
  <|im_start|>user
  Answer the following question...<|im_end|>
  <|im_start|>assistant
  """
)

# Chat messages automatically handle templating
llm([
  {"role": "system", "content": "You are a friendly assistant."},
  {"role": "user", "content": "Answer the following question..."}
])

# Set the default role to user and string inputs are converted to chat messages
llm("Answer the following question...", defaultrole="user")
```

The LLM pipeline automatically detects the underlying LLM framework. This can also be manually set.

[Hugging Face Transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/abetlen/llama-cpp-python) and [hosted API models via LiteLLM](https://github.com/BerriAI/litellm) are all supported by this pipeline.

See the [LiteLLM documentation](https://litellm.vercel.app/docs/providers) for the options available with LiteLLM models. llama.cpp models support both local and remote GGUF paths on the HF Hub.

```python
from txtai import LLM

# Transformers
llm = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct")
llm = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", method="transformers")

# llama.cpp
llm = LLM("microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf")
llm = LLM("microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
           method="llama.cpp")

# LiteLLM
llm = LLM("ollama/llama3.1")
llm = LLM("ollama/llama3.1", method="litellm")

# Custom Ollama endpoint
llm = LLM("ollama/llama3.1", api_base="http://localhost:11434")

# Custom OpenAI-compatible endpoint
llm = LLM("openai/llama3.1", api_base="http://localhost:4000")

# LLM APIs - must also set API key via environment variable
llm = LLM("gpt-4o")
llm = LLM("claude-3-5-sonnet-20240620")
```

Models can be externally loaded and passed to pipelines. This is useful for models that are not yet supported by Transformers and/or need special initialization.

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from txtai import LLM

# Load Phi 3.5-mini
path = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
  path,
  torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(path)

llm = LLM((model, tokenizer))
```

See the links below for more detailed examples.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Prompt-driven search with LLMs](https://github.com/neuml/txtai/blob/master/examples/42_Prompt_driven_search_with_LLMs.ipynb) | Embeddings-guided and Prompt-driven search with Large Language Models (LLMs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/42_Prompt_driven_search_with_LLMs.ipynb) |
| [Prompt templates and task chains](https://github.com/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) | Build model prompts and connect tasks together with workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) |
| [Build RAG pipelines with txtai](https://github.com/neuml/txtai/blob/master/examples/52_Build_RAG_pipelines_with_txtai.ipynb) | Guide on retrieval augmented generation including how to create citations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/52_Build_RAG_pipelines_with_txtai.ipynb) |
| [Integrate LLM frameworks](https://github.com/neuml/txtai/blob/master/examples/53_Integrate_LLM_Frameworks.ipynb) | Integrate llama.cpp, LiteLLM and custom generation frameworks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/53_Integrate_LLM_Frameworks.ipynb) |
| [Generate knowledge with Semantic Graphs and RAG](https://github.com/neuml/txtai/blob/master/examples/55_Generate_knowledge_with_Semantic_Graphs_and_RAG.ipynb) | Knowledge exploration and discovery with Semantic Graphs and RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/55_Generate_knowledge_with_Semantic_Graphs_and_RAG.ipynb) |
| [Build knowledge graphs with LLMs](https://github.com/neuml/txtai/blob/master/examples/57_Build_knowledge_graphs_with_LLM_driven_entity_extraction.ipynb) | Build knowledge graphs with LLM-driven entity extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/57_Build_knowledge_graphs_with_LLM_driven_entity_extraction.ipynb) |
| [Advanced RAG with graph path traversal](https://github.com/neuml/txtai/blob/master/examples/58_Advanced_RAG_with_graph_path_traversal.ipynb) | Graph path traversal to collect complex sets of data for advanced RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/58_Advanced_RAG_with_graph_path_traversal.ipynb) |
| [Advanced RAG with guided generation](https://github.com/neuml/txtai/blob/master/examples/60_Advanced_RAG_with_guided_generation.ipynb) | Retrieval Augmented and Guided Generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/60_Advanced_RAG_with_guided_generation.ipynb) |
| [RAG with llama.cpp and external API services](https://github.com/neuml/txtai/blob/master/examples/62_RAG_with_llama_cpp_and_external_API_services.ipynb) | RAG with additional vector and LLM frameworks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/62_RAG_with_llama_cpp_and_external_API_services.ipynb) |
| [How RAG with txtai works](https://github.com/neuml/txtai/blob/master/examples/63_How_RAG_with_txtai_works.ipynb) | Create RAG processes, API services and Docker instances | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/63_How_RAG_with_txtai_works.ipynb) |
| [Speech to Speech RAG](https://github.com/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) [▶️](https://www.youtube.com/watch?v=tH8QWwkVMKA) | Full cycle speech to speech workflow with RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) |
| [Generative Audio](https://github.com/neuml/txtai/blob/master/examples/66_Generative_Audio.ipynb) | Storytelling with generative audio workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/66_Generative_Audio.ipynb) |
| [Analyzing Hugging Face Posts with Graphs and Agents](https://github.com/neuml/txtai/blob/master/examples/68_Analyzing_Hugging_Face_Posts_with_Graphs_and_Agents.ipynb) | Explore a rich dataset with Graph Analysis and Agents | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/68_Analyzing_Hugging_Face_Posts_with_Graphs_and_Agents.ipynb) |
| [Granting autonomy to agents](https://github.com/neuml/txtai/blob/master/examples/69_Granting_autonomy_to_agents.ipynb) | Agents that iteratively solve problems as they see fit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/69_Granting_autonomy_to_agents.ipynb) |
| [Getting started with LLM APIs](https://github.com/neuml/txtai/blob/master/examples/70_Getting_started_with_LLM_APIs.ipynb) | Generate embeddings and run LLMs with OpenAI, Claude, Gemini, Bedrock and more | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/70_Getting_started_with_LLM_APIs.ipynb) |
| [Analyzing LinkedIn Company Posts with Graphs and Agents](https://github.com/neuml/txtai/blob/master/examples/71_Analyzing_LinkedIn_Company_Posts_with_Graphs_and_Agents.ipynb) | Exploring how to improve social media engagement with AI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/71_Analyzing_LinkedIn_Company_Posts_with_Graphs_and_Agents.ipynb) |
| [Parsing the stars with txtai](https://github.com/neuml/txtai/blob/master/examples/72_Parsing_the_stars_with_txtai.ipynb) | Explore an astronomical knowledge graph of known stars, planets, galaxies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/72_Parsing_the_stars_with_txtai.ipynb) |
| [Chunking your data for RAG](https://github.com/neuml/txtai/blob/master/examples/73_Chunking_your_data_for_RAG.ipynb) | Extract, chunk and index content for effective retrieval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/73_Chunking_your_data_for_RAG.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
llm:

# Run pipeline with workflow
workflow:
  llm:
    tasks:
      - action: llm
```

Similar to the Python example above, the underlying [Hugging Face pipeline parameters](https://huggingface.co/docs/transformers/main/main_classes/pipelines#transformers.pipeline.model) and [model parameters](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel.from_pretrained) can be set in pipeline configuration.

```yaml
llm:
  path: microsoft/Phi-3.5-mini-instruct
  torch_dtype: torch.bfloat16
```

### Run with Workflows

```python
from txtai import Application

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
  -d '{"name":"llm", "elements": ["Answer the following question..."]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.LLM.__init__
### ::: txtai.pipeline.LLM.__call__

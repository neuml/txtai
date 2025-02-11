# RAG

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The RAG pipeline (aka Extractor) joins a prompt, context data store and generative model together to extract knowledge.

The data store can be an embeddings database or a similarity instance with associated input text. The generative model can be a prompt-driven large language model (LLM), an extractive question-answering model or a custom pipeline. This is known as retrieval augmented generation (RAG).

## Example

The following shows a simple example using this pipeline.

```python
from txtai import Embeddings, RAG

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
embeddings = Embeddings(content=True)
embeddings.index(data)

# Create and run pipeline
rag = RAG(embeddings, "google/flan-t5-base", template="""
  Answer the following question using the provided context.

  Question:
  {question}

  Context:
  {context}
""")

rag("What was won?")

# Instruction tuned models typically require string prompts to
# follow a specific chat template set by the model
rag = RAG(embeddings, "meta-llama/Meta-Llama-3.1-8B-Instruct", template="""
  <|im_start|>system
  You are a friendly assistant.<|im_end|>
  <|im_start|>user
  Answer the following question using the provided context.

  Question:
  {question}

  Context:
  {context}
  <|im_start|>assistant
  """
)
rag("What was won?")

# LLM options can be passed as additional arguments
rag = RAG(embeddings, "meta-llama/Meta-Llama-3.1-8B-Instruct", template="""
  Answer the following question using the provided context.

  Question:
  {question}

  Context:
  {context}
""")

# Set the default role to user and string inputs are converted to chat messages
rag("What was won?", defaultrole="user")
```

See the [Embeddings](../../../embeddings) and [LLM](../llm) pages for additional configuration options.

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
| [Extractive QA with txtai](https://github.com/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) | Introduction to extractive question-answering with txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) |
| [Extractive QA with Elasticsearch](https://github.com/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) | Run extractive question-answering queries with Elasticsearch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) |
| [Extractive QA to build structured data](https://github.com/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) | Build structured datasets using extractive question-answering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) |
| [Parsing the stars with txtai](https://github.com/neuml/txtai/blob/master/examples/72_Parsing_the_stars_with_txtai.ipynb) | Explore an astronomical knowledge graph of known stars, planets, galaxies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/72_Parsing_the_stars_with_txtai.ipynb) |
| [Chunking your data for RAG](https://github.com/neuml/txtai/blob/master/examples/73_Chunking_your_data_for_RAG.ipynb) | Extract, chunk and index content for effective retrieval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/73_Chunking_your_data_for_RAG.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Allow documents to be indexed
writable: True

# Content is required for extractor pipeline
embeddings:
  content: True

rag:
  path: google/flan-t5-base
  template: |
    Answer the following question using the provided context.

    Question:
    {question}

    Context:
    {context}

workflow:
  search:
    tasks:
      - action: rag
```

### Run with Workflows

Built in tasks make using the extractor pipeline easier.

```python
from txtai import Application

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

### ::: txtai.pipeline.RAG.__init__
### ::: txtai.pipeline.RAG.__call__

# Use Cases

The following sections introduce common txtai use cases. A comprehensive set of over 60 [example notebooks and applications](../examples) are also available.

## Semantic Search

Build semantic/similarity/vector/neural search applications.

![demo](https://raw.githubusercontent.com/neuml/txtai/master/demo.gif)

Traditional search systems use keywords to find data. Semantic search has an understanding of natural language and identifies results that have the same meaning, not necessarily the same keywords.

![search](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/search.png#gh-light-mode-only)
![search](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/search-dark.png#gh-dark-mode-only)

Get started with the following examples.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Introducing txtai](https://github.com/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) [▶️](https://www.youtube.com/watch?v=SIezMnVdmMs) | Overview of the functionality provided by txtai | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) |
| [Similarity search with images](https://github.com/neuml/txtai/blob/master/examples/13_Similarity_search_with_images.ipynb) | Embed images and text into the same space for search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/13_Similarity_search_with_images.ipynb) |
| [Build a QA database](https://github.com/neuml/txtai/blob/master/examples/34_Build_a_QA_database.ipynb) | Question matching with semantic search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/34_Build_a_QA_database.ipynb) |
| [Semantic Graphs](https://github.com/neuml/txtai/blob/master/examples/38_Introducing_the_Semantic_Graph.ipynb) | Explore topics, data connectivity and run network analysis| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/38_Introducing_the_Semantic_Graph.ipynb) |

## LLM Orchestration

Autonomous agents, retrieval augmented generation (RAG), chat with your data, pipelines and workflows that interface with large language models (LLMs).

![llm](images/llm.png)

See below to learn more.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Prompt templates and task chains](https://github.com/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) | Build model prompts and connect tasks together with workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) |
| [Integrate LLM frameworks](https://github.com/neuml/txtai/blob/master/examples/53_Integrate_LLM_Frameworks.ipynb) | Integrate llama.cpp, LiteLLM and custom generation frameworks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/53_Integrate_LLM_Frameworks.ipynb) |
| [Build knowledge graphs with LLMs](https://github.com/neuml/txtai/blob/master/examples/57_Build_knowledge_graphs_with_LLM_driven_entity_extraction.ipynb) | Build knowledge graphs with LLM-driven entity extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/57_Build_knowledge_graphs_with_LLM_driven_entity_extraction.ipynb) |

### Agents

Agents connect embeddings, pipelines, workflows and other agents together to autonomously solve complex problems.

![agent](images/agent.png)

txtai agents are built on top of the [smolagents](https://github.com/huggingface/smolagents) framework. This supports all LLMs txtai supports (Hugging Face, llama.cpp, OpenAI / Claude / AWS Bedrock via LiteLLM).

See the link below to learn more.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Analyzing Hugging Face Posts with Graphs and Agents](https://github.com/neuml/txtai/blob/master/examples/68_Analyzing_Hugging_Face_Posts_with_Graphs_and_Agents.ipynb) | Explore a rich dataset with Graph Analysis and Agents | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/68_Analyzing_Hugging_Face_Posts_with_Graphs_and_Agents.ipynb) |
| [Granting autonomy to agents](https://github.com/neuml/txtai/blob/master/examples/69_Granting_autonomy_to_agents.ipynb) | Agents that iteratively solve problems as they see fit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/69_Granting_autonomy_to_agents.ipynb) |
| [Analyzing LinkedIn Company Posts with Graphs and Agents](https://github.com/neuml/txtai/blob/master/examples/71_Analyzing_LinkedIn_Company_Posts_with_Graphs_and_Agents.ipynb) | Exploring how to improve social media engagement with AI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/71_Analyzing_LinkedIn_Company_Posts_with_Graphs_and_Agents.ipynb) |

### Retrieval augmented generation

Retrieval augmented generation (RAG) reduces the risk of LLM hallucinations by constraining the output with a knowledge base as context. RAG is commonly used to "chat with your data".

![rag](images/rag.png#gh-light-mode-only)
![rag](images/rag-dark.png#gh-dark-mode-only)

A novel feature of txtai is that it can provide both an answer and source citation.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Build RAG pipelines with txtai](https://github.com/neuml/txtai/blob/master/examples/52_Build_RAG_pipelines_with_txtai.ipynb) | Guide on retrieval augmented generation including how to create citations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/52_Build_RAG_pipelines_with_txtai.ipynb) |
| [How RAG with txtai works](https://github.com/neuml/txtai/blob/master/examples/63_How_RAG_with_txtai_works.ipynb) | Create RAG processes, API services and Docker instances | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/63_How_RAG_with_txtai_works.ipynb) |
| [Advanced RAG with graph path traversal](https://github.com/neuml/txtai/blob/master/examples/58_Advanced_RAG_with_graph_path_traversal.ipynb) | Graph path traversal to collect complex sets of data for advanced RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/58_Advanced_RAG_with_graph_path_traversal.ipynb) |
| [Speech to Speech RAG](https://github.com/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) [▶️](https://www.youtube.com/watch?v=tH8QWwkVMKA) | Full cycle speech to speech workflow with RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) |

## Language Model Workflows

Language model workflows, also known as semantic workflows, connect language models together to build intelligent applications.

![flows](images/flows.png#gh-light-mode-only)
![flows](images/flows-dark.png#gh-dark-mode-only)

While LLMs are powerful, there are plenty of smaller, more specialized models that work better and faster for specific tasks. This includes models for extractive question-answering, automatic summarization, text-to-speech, transcription and translation.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Run pipeline workflows](https://github.com/neuml/txtai/blob/master/examples/14_Run_pipeline_workflows.ipynb) [▶️](https://www.youtube.com/watch?v=UBMPDCn1gEU) | Simple yet powerful constructs to efficiently process data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/14_Run_pipeline_workflows.ipynb) |
| [Building abstractive text summaries](https://github.com/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) | Run abstractive text summarization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) |
| [Transcribe audio to text](https://github.com/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) | Convert audio files to text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) |
| [Translate text between languages](https://github.com/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) | Streamline machine translation and language detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) |

<p align="center">
    <img src="https://raw.githubusercontent.com/neuml/txtai/master/logo.png"/>
</p>

<p align="center">
    <b>All-in-one AI framework</b>
</p>

<p align="center">
    <a href="https://github.com/neuml/txtai/releases">
        <img src="https://img.shields.io/github/release/neuml/txtai.svg?style=flat&color=success" alt="Version"/>
    </a>
    <a href="https://github.com/neuml/txtai">
        <img src="https://img.shields.io/github/last-commit/neuml/txtai.svg?style=flat&color=blue" alt="GitHub last commit"/>
    </a>
    <a href="https://github.com/neuml/txtai/issues">
        <img src="https://img.shields.io/github/issues/neuml/txtai.svg?style=flat&color=success" alt="GitHub issues"/>
    </a>
    <a href="https://join.slack.com/t/txtai/shared_invite/zt-37c1zfijp-Y57wMty6YOx_hyIHEQvQJA">
        <img src="https://img.shields.io/badge/slack-join-blue?style=flat&logo=slack&logocolor=white" alt="Join Slack"/>
    </a>
    <a href="https://github.com/neuml/txtai/actions?query=workflow%3Abuild">
        <img src="https://github.com/neuml/txtai/workflows/build/badge.svg" alt="Build Status"/>
    </a>
    <a href="https://coveralls.io/github/neuml/txtai?branch=master">
        <img src="https://img.shields.io/coverallsCoverage/github/neuml/txtai" alt="Coverage Status">
    </a>
</p>

txtai is an all-in-one AI framework for semantic search, LLM orchestration and language model workflows.

![architecture](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/architecture.png#gh-light-mode-only)
![architecture](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/architecture-dark.png#gh-dark-mode-only)

The key component of txtai is an embeddings database, which is a union of vector indexes (sparse and dense), graph networks and relational databases.

This foundation enables vector search and/or serves as a powerful knowledge source for large language model (LLM) applications.

Build autonomous agents, retrieval augmented generation (RAG) processes, multi-model workflows and more.

Summary of txtai features:

- 🔎 Vector search with SQL, object storage, topic modeling, graph analysis and multimodal indexing
- 📄 Create embeddings for text, documents, audio, images and video
- 💡 Pipelines powered by language models that run LLM prompts, question-answering, labeling, transcription, translation, summarization and more
- ↪️️ Workflows to join pipelines together and aggregate business logic. txtai processes can be simple microservices or multi-model workflows.
- 🤖 Agents that intelligently connect embeddings, pipelines, workflows and other agents together to autonomously solve complex problems
- ⚙️ Web and Model Context Protocol (MCP) APIs. Bindings available for [JavaScript](https://github.com/neuml/txtai.js), [Java](https://github.com/neuml/txtai.java), [Rust](https://github.com/neuml/txtai.rs) and [Go](https://github.com/neuml/txtai.go).
- 🔋 Batteries included with defaults to get up and running fast
- ☁️ Run local or scale out with container orchestration

txtai is built with Python 3.10+, [Hugging Face Transformers](https://github.com/huggingface/transformers), [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and [FastAPI](https://github.com/tiangolo/fastapi). txtai is open-source under an Apache 2.0 license.

*Interested in an easy and secure way to run hosted txtai applications? Then join the [txtai.cloud](https://txtai.cloud) preview to learn more.*

## Why txtai?

![why](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/why.png#gh-light-mode-only)
![why](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/why-dark.png#gh-dark-mode-only)

New vector databases, LLM frameworks and everything in between are sprouting up daily. Why build with txtai?

- Up and running in minutes with [pip](https://neuml.github.io/txtai/install/) or [Docker](https://neuml.github.io/txtai/cloud/)
```python
# Get started in a couple lines
import txtai

embeddings = txtai.Embeddings()
embeddings.index(["Correct", "Not what we hoped"])
embeddings.search("positive", 1)
#[(0, 0.29862046241760254)]
```
- Built-in API makes it easy to develop applications using your programming language of choice
```yaml
# app.yml
embeddings:
    path: sentence-transformers/all-MiniLM-L6-v2
```
```bash
CONFIG=app.yml uvicorn "txtai.api:app"
curl -X GET "http://localhost:8000/search?query=positive"
```
- Run local - no need to ship data off to disparate remote services
- Work with micromodels all the way up to large language models (LLMs)
- Low footprint - install additional dependencies and scale up when needed
- [Learn by example](https://neuml.github.io/txtai/examples) - notebooks cover all available functionality

## Use Cases

The following sections introduce common txtai use cases. A comprehensive set of over 60 [example notebooks and applications](https://neuml.github.io/txtai/examples) are also available.

### Semantic Search

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

### LLM Orchestration

Autonomous agents, retrieval augmented generation (RAG), chat with your data, pipelines and workflows that interface with large language models (LLMs).

![llm](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/llm.png)

See below to learn more.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Prompt templates and task chains](https://github.com/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) | Build model prompts and connect tasks together with workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/44_Prompt_templates_and_task_chains.ipynb) |
| [Integrate LLM frameworks](https://github.com/neuml/txtai/blob/master/examples/53_Integrate_LLM_Frameworks.ipynb) | Integrate llama.cpp, LiteLLM and custom generation frameworks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/53_Integrate_LLM_Frameworks.ipynb) |
| [Build knowledge graphs with LLMs](https://github.com/neuml/txtai/blob/master/examples/57_Build_knowledge_graphs_with_LLM_driven_entity_extraction.ipynb) | Build knowledge graphs with LLM-driven entity extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/57_Build_knowledge_graphs_with_LLM_driven_entity_extraction.ipynb) |
| [Parsing the stars with txtai](https://github.com/neuml/txtai/blob/master/examples/72_Parsing_the_stars_with_txtai.ipynb) | Explore an astronomical knowledge graph of known stars, planets, galaxies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/72_Parsing_the_stars_with_txtai.ipynb) |

#### Agents

Agents connect embeddings, pipelines, workflows and other agents together to autonomously solve complex problems.

![agent](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/agent.png)

txtai agents are built on top of the [smolagents](https://github.com/huggingface/smolagents) framework. This supports all LLMs txtai supports (Hugging Face, llama.cpp, OpenAI / Claude / AWS Bedrock via LiteLLM).

See the link below to learn more.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Analyzing Hugging Face Posts with Graphs and Agents](https://github.com/neuml/txtai/blob/master/examples/68_Analyzing_Hugging_Face_Posts_with_Graphs_and_Agents.ipynb) | Explore a rich dataset with Graph Analysis and Agents | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/68_Analyzing_Hugging_Face_Posts_with_Graphs_and_Agents.ipynb) |
| [Granting autonomy to agents](https://github.com/neuml/txtai/blob/master/examples/69_Granting_autonomy_to_agents.ipynb) | Agents that iteratively solve problems as they see fit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/69_Granting_autonomy_to_agents.ipynb) |
| [Analyzing LinkedIn Company Posts with Graphs and Agents](https://github.com/neuml/txtai/blob/master/examples/71_Analyzing_LinkedIn_Company_Posts_with_Graphs_and_Agents.ipynb) | Exploring how to improve social media engagement with AI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/71_Analyzing_LinkedIn_Company_Posts_with_Graphs_and_Agents.ipynb) |

#### Retrieval augmented generation

Retrieval augmented generation (RAG) reduces the risk of LLM hallucinations by constraining the output with a knowledge base as context. RAG is commonly used to "chat with your data".

![rag](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/rag.png#gh-light-mode-only)
![rag](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/rag-dark.png#gh-dark-mode-only)

A novel feature of txtai is that it can provide both an answer and source citation.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Build RAG pipelines with txtai](https://github.com/neuml/txtai/blob/master/examples/52_Build_RAG_pipelines_with_txtai.ipynb) | Guide on retrieval augmented generation including how to create citations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/52_Build_RAG_pipelines_with_txtai.ipynb) |
| [Chunking your data for RAG](https://github.com/neuml/txtai/blob/master/examples/73_Chunking_your_data_for_RAG.ipynb) | Extract, chunk and index content for effective retrieval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/73_Chunking_your_data_for_RAG.ipynb) |
| [Advanced RAG with graph path traversal](https://github.com/neuml/txtai/blob/master/examples/58_Advanced_RAG_with_graph_path_traversal.ipynb) | Graph path traversal to collect complex sets of data for advanced RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/58_Advanced_RAG_with_graph_path_traversal.ipynb) |
| [Speech to Speech RAG](https://github.com/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) [▶️](https://www.youtube.com/watch?v=tH8QWwkVMKA) | Full cycle speech to speech workflow with RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/65_Speech_to_Speech_RAG.ipynb) |

### Language Model Workflows

Language model workflows, also known as semantic workflows, connect language models together to build intelligent applications.

![flows](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/flows.png#gh-light-mode-only)
![flows](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/flows-dark.png#gh-dark-mode-only)

While LLMs are powerful, there are plenty of smaller, more specialized models that work better and faster for specific tasks. This includes models for extractive question-answering, automatic summarization, text-to-speech, transcription and translation.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Run pipeline workflows](https://github.com/neuml/txtai/blob/master/examples/14_Run_pipeline_workflows.ipynb) [▶️](https://www.youtube.com/watch?v=UBMPDCn1gEU) | Simple yet powerful constructs to efficiently process data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/14_Run_pipeline_workflows.ipynb) |
| [Building abstractive text summaries](https://github.com/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) | Run abstractive text summarization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) |
| [Transcribe audio to text](https://github.com/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) | Convert audio files to text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) |
| [Translate text between languages](https://github.com/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) | Streamline machine translation and language detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) |

## Installation

![install](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/install.png#gh-light-mode-only)
![install](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/install-dark.png#gh-dark-mode-only)

The easiest way to install is via pip and PyPI

```
pip install txtai
```

Python 3.10+ is supported. Using a Python [virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

See the detailed [install instructions](https://neuml.github.io/txtai/install) for more information covering [optional dependencies](https://neuml.github.io/txtai/install/#optional-dependencies), [environment specific prerequisites](https://neuml.github.io/txtai/install/#environment-specific-prerequisites), [installing from source](https://neuml.github.io/txtai/install/#install-from-source), [conda support](https://neuml.github.io/txtai/install/#conda) and how to [run with containers](https://neuml.github.io/txtai/cloud).

## Model guide

![models](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/models.png)

See the table below for the current recommended models. These models all allow commercial use and offer a blend of speed and performance.

| Component                                                                     | Model(s)                                                                 |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Embeddings](https://neuml.github.io/txtai/embeddings)                        | [all-MiniLM-L6-v2](https://hf.co/sentence-transformers/all-MiniLM-L6-v2) | 
| [Image Captions](https://neuml.github.io/txtai/pipeline/image/caption)        | [BLIP](https://hf.co/Salesforce/blip-image-captioning-base)              |
| [Labels - Zero Shot](https://neuml.github.io/txtai/pipeline/text/labels)      | [BART-Large-MNLI](https://hf.co/facebook/bart-large)                     |
| [Labels - Fixed](https://neuml.github.io/txtai/pipeline/text/labels)          | Fine-tune with [training pipeline](https://neuml.github.io/txtai/pipeline/train/trainer)          |
| [Large Language Model (LLM)](https://neuml.github.io/txtai/pipeline/text/llm) | [Llama 3.1 Instruct](https://hf.co/meta-llama/Llama-3.1-8B-Instruct)     |
| [Summarization](https://neuml.github.io/txtai/pipeline/text/summary)          | [DistilBART](https://hf.co/sshleifer/distilbart-cnn-12-6)                |
| [Text-to-Speech](https://neuml.github.io/txtai/pipeline/audio/texttospeech)   | [ESPnet JETS](https://hf.co/NeuML/ljspeech-jets-onnx)                    |
| [Transcription](https://neuml.github.io/txtai/pipeline/audio/transcription)   | [Whisper](https://hf.co/openai/whisper-base)                             | 
| [Translation](https://neuml.github.io/txtai/pipeline/text/translation)        | [OPUS Model Series](https://hf.co/Helsinki-NLP)                          |

Models can be loaded as either a path from the Hugging Face Hub or a local directory. Model paths are optional, defaults are loaded when not specified. For tasks with no recommended model, txtai uses the default models as shown in the Hugging Face Tasks guide.

See the following links to learn more.

- [Hugging Face Tasks](https://hf.co/tasks)
- [Hugging Face Model Hub](https://hf.co/models)
- [MTEB Leaderboard](https://hf.co/spaces/mteb/leaderboard)
- [LMSYS LLM Leaderboard](https://chat.lmsys.org/?leaderboard)
- [Open LLM Leaderboard](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard)

## Powered by txtai

The following applications are powered by txtai.

![apps](https://raw.githubusercontent.com/neuml/txtai/master/apps.jpg)

| Application  | Description  |
|:------------ |:-------------|
| [rag](https://github.com/neuml/rag) | Retrieval Augmented Generation (RAG) application |
| [ragdata](https://github.com/neuml/ragdata) | Build knowledge bases for RAG |
| [paperai](https://github.com/neuml/paperai) | Semantic search and workflows for medical/scientific papers |
| [annotateai](https://github.com/neuml/annotateai) | Automatically annotate papers with LLMs |

In addition to this list, there are also many other [open-source projects](https://github.com/neuml/txtai/network/dependents), [published research](https://scholar.google.com/scholar?q=txtai&hl=en&as_ylo=2022) and closed proprietary/commercial projects that have built on txtai in production.

## Further Reading

![further](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/further.png#gh-light-mode-only)
![further](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/further-ghdark.png#gh-dark-mode-only)

- [Introducing txtai, the all-in-one AI framework](https://medium.com/neuml/introducing-txtai-the-all-in-one-ai-framework-0660ecfc39d7)
- [Tutorial series on Hashnode](https://neuml.hashnode.dev/series/txtai-tutorial) | [dev.to](https://dev.to/neuml/tutorial-series-on-txtai-ibg)
- [What's new in txtai 8.0](https://medium.com/neuml/whats-new-in-txtai-8-0-2d7d0ab4506b) | [7.0](https://medium.com/neuml/whats-new-in-txtai-7-0-855ad6a55440) | [6.0](https://medium.com/neuml/whats-new-in-txtai-6-0-7d93eeedf804) | [5.0](https://medium.com/neuml/whats-new-in-txtai-5-0-e5c75a13b101) | [4.0](https://medium.com/neuml/whats-new-in-txtai-4-0-bbc3a65c3d1c)
- [Getting started with semantic search](https://medium.com/neuml/getting-started-with-semantic-search-a9fd9d8a48cf) | [workflows](https://medium.com/neuml/getting-started-with-semantic-workflows-2fefda6165d9) | [rag](https://medium.com/neuml/getting-started-with-rag-9a0cca75f748)
- [Running txtai at scale](https://medium.com/neuml/running-at-scale-with-txtai-71196cdd99f9)
- [Vector search & RAG Landscape: A review with txtai](https://medium.com/neuml/vector-search-rag-landscape-a-review-with-txtai-a7f37ad0e187)

## Documentation

[Full documentation on txtai](https://neuml.github.io/txtai) including configuration settings for embeddings, pipelines, workflows, API and a FAQ with common questions/issues is available.

## Contributing

For those who would like to contribute to txtai, please see [this guide](https://github.com/neuml/.github/blob/master/CONTRIBUTING.md).

#

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

![architecture](images/architecture.png#gh-light-mode-only)
![architecture](images/architecture-dark.png#gh-dark-mode-only)

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

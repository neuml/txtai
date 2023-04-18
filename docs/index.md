#

<p align="center">
    <img src="https://raw.githubusercontent.com/neuml/txtai/master/logo.png"/>
</p>

<h3 align="center">
    <p>Semantic search and workflows powered by language models</p>
</h3>

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
    <a href="https://join.slack.com/t/txtai/shared_invite/zt-1cagya4yf-DQeuZbd~aMwH5pckBU4vPg">
        <img src="https://img.shields.io/badge/slack-join-blue?style=flat&logo=slack&logocolor=white" alt="Join Slack"/>
    </a>
    <a href="https://github.com/neuml/txtai/actions?query=workflow%3Abuild">
        <img src="https://github.com/neuml/txtai/workflows/build/badge.svg" alt="Build Status"/>
    </a>
    <a href="https://coveralls.io/github/neuml/txtai?branch=master">
        <img src="https://img.shields.io/coverallsCoverage/github/neuml/txtai" alt="Coverage Status">
    </a>
</p>

-------------------------------------------------------------------------------------------------------------------------------------------------------

txtai is an open-source platform for semantic search and workflows powered by language models.

![demo](https://raw.githubusercontent.com/neuml/txtai/master/demo.gif)

Traditional search systems use keywords to find data. Semantic search has an understanding of natural language and identifies results that have the same meaning, not necessarily the same keywords.

![search](images/search.png#only-light)
![search](images/search-dark.png#only-dark)

txtai builds embeddings databases, which are a union of vector indexes and relational databases. This enables similarity search with SQL. Embeddings databases can stand on their own and/or serve as a powerful knowledge source for large language model (LLM) prompts.

Semantic workflows connect language models together to build intelligent applications.

![flows](images/flows.png#only-light)
![flows](images/flows-dark.png#only-dark)

Integrate vector search, conversational search, automatic summarization, transcription, translation and more.

Summary of txtai features:

- 🔎 Similarity search with SQL, object storage, topic modeling, graph analysis, multiple vector index backends ([Faiss](https://github.com/facebookresearch/faiss), [Annoy](https://github.com/spotify/annoy), [Hnswlib](https://github.com/nmslib/hnswlib)) and support for external vector databases
- 📄 Create embeddings for text, documents, audio, images and video
- 💡 Pipelines powered by language models that run question-answering, labeling, transcription, translation, summarization, LLM prompts and more
- ↪️️ Workflows to join pipelines together and aggregate business logic. txtai processes can be simple microservices or multi-model workflows.
- ⚙️ Build with Python or YAML. API bindings available for [JavaScript](https://github.com/neuml/txtai.js), [Java](https://github.com/neuml/txtai.java), [Rust](https://github.com/neuml/txtai.rs) and [Go](https://github.com/neuml/txtai.go).
- ☁️ Cloud-native architecture that scales out with container orchestration systems (e.g. Kubernetes)

The following applications are powered by txtai.

![apps](https://raw.githubusercontent.com/neuml/txtai/master/apps.jpg)

| Application  | Description  |
|:----------|:-------------|
| [txtchat](https://github.com/neuml/txtchat) | Conversational search and workflows for all |
| [paperai](https://github.com/neuml/paperai) | Semantic search and workflows for medical/scientific papers |
| [codequestion](https://github.com/neuml/codequestion) | Semantic search for developers |
| [tldrstory](https://github.com/neuml/tldrstory) | Semantic search for headlines and story text |

txtai is built with Python 3.7+, [Hugging Face Transformers](https://github.com/huggingface/transformers), [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and [FastAPI](https://github.com/tiangolo/fastapi)

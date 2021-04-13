#

<p align="center">
    <img src="https://raw.githubusercontent.com/neuml/txtai/master/logo.png"/>
</p>

<h3 align="center">
    <p>AI-powered search engine</p>
</h3>

<p align="center">
    <a href="https://github.com/neuml/txtai/releases">
        <img src="https://img.shields.io/github/release/neuml/txtai.svg?style=flat&color=success" alt="Version"/>
    </a>
    <a href="https://github.com/neuml/txtai/releases">
        <img src="https://img.shields.io/github/release-date/neuml/txtai.svg?style=flat&color=blue" alt="GitHub Release Date"/>
    </a>
    <a href="https://github.com/neuml/txtai/issues">
        <img src="https://img.shields.io/github/issues/neuml/txtai.svg?style=flat&color=success" alt="GitHub issues"/>
    </a>
    <a href="https://github.com/neuml/txtai">
        <img src="https://img.shields.io/github/last-commit/neuml/txtai.svg?style=flat&color=blue" alt="GitHub last commit"/>
    </a>
    <a href="https://github.com/neuml/txtai/actions?query=workflow%3Abuild">
        <img src="https://github.com/neuml/txtai/workflows/build/badge.svg" alt="Build Status"/>
    </a>
    <a href="https://coveralls.io/github/neuml/txtai?branch=master">
        <img src="https://img.shields.io/coveralls/github/neuml/txtai" alt="Coverage Status">
    </a>
</p>

-------------------------------------------------------------------------------------------------------------------------------------------------------

txtai executes machine-learning workflows to transform data and build AI-powered text indices to perform similarity search.

![demo](https://raw.githubusercontent.com/neuml/txtai/master/demo.gif)

Summary of txtai features:

- 🔎 Large-scale similarity search with multiple index backends ([Faiss](https://github.com/facebookresearch/faiss), [Annoy](https://github.com/spotify/annoy), [Hnswlib](https://github.com/nmslib/hnswlib))
- 📄 Create embeddings for text snippets, documents, audio and images. Supports transformers and word vectors.
- 💡 Machine-learning pipelines to run extractive question-answering, zero-shot labeling, transcription, translation, summarization and text extraction
- ↪️️ Workflows that join pipelines together to aggregate business logic. txtai processes can be microservices or full-fledged indexing workflows.
- 🔗 API bindings for [JavaScript](https://github.com/neuml/txtai.js), [Java](https://github.com/neuml/txtai.java), [Rust](https://github.com/neuml/txtai.rs) and [Go](https://github.com/neuml/txtai.go)
- ☁️ Cloud-native architecture that scales out with container orchestration systems (e.g. Kubernetes)

txtai and/or the concepts behind it has already been used to power the Natural Language Processing (NLP) applications listed below:

| Application  | Description  |
|:----------|:-------------|
| [paperai](https://github.com/neuml/paperai) | AI-powered literature discovery and review engine for medical/scientific papers |
| [tldrstory](https://github.com/neuml/tldrstory) | AI-powered understanding of headlines and story text |
| [neuspo](https://neuspo.com) | Fact-driven, real-time sports event and news site |
| [codequestion](https://github.com/neuml/codequestion) | Ask coding questions directly from the terminal |

txtai is built with Python 3.6+, [Hugging Face Transformers](https://github.com/huggingface/transformers), [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and [FastAPI](https://github.com/tiangolo/fastapi)

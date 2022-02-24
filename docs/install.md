# Installation

![install](images/install.png#only-light)
![install](images/install-dark.png#only-dark)

The easiest way to install is via pip and PyPI

    pip install txtai

Python 3.7+ is supported. Using a Python [virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

## Install from source

txtai can also be installed directly from GitHub to access the latest, unreleased features.

    pip install git+https://github.com/neuml/txtai

## Environment specific prerequisites

Additional environment specific prerequisites are below.

### Linux

Optional audio transcription requires a [system library to be installed](https://github.com/bastibe/python-soundfile#installation)

### macOS

Run `brew install libomp` see [this link](https://github.com/kyamagu/faiss-wheels#prerequisite)

### Windows

Optional dependencies require [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Optional dependencies

txtai has the following optional dependencies that can be installed as extras. The patterns below are supported
in setup.py install_requires sections.

### All

Install all dependencies (mirrors txtai < 3.2)

```
pip install txtai[all]
```

### API

Serve txtai via a web API.

```
pip install txtai[api]
```

### Cloud

Interface with cloud compute.

```
pip install txtai[cloud]
```

### Database

Additional content storage options

```
pip install txtai[database]
```

### Model

Additional non-standard models

```
pip install txtai[model]
```

### Pipeline

All pipelines - default install comes with most common pipelines.

```
pip install txtai[pipeline]
```

### Similarity

Word vectors, support for sentence-transformers models not on the HF Hub and additional ANN libraries.

```
pip install txtai[similarity]
```

### Workflow

All workflow tasks - default install comes with most common workflow tasks.

```
pip install txtai[workflow]
```

Multiple dependencies can be specified at the same time.

```
pip install txtai[pipeline,workflow]
```

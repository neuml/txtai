# Installation

![install](images/install.png#only-light)
![install](images/install-dark.png#only-dark)

The easiest way to install is via pip and PyPI

```
pip install txtai
```

Python 3.10+ is supported. Using a Python [virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

## Optional dependencies

txtai has the following optional dependencies that can be installed as extras. The patterns below are supported
in setup.py install_requires sections.

_Note: Extras are provided for convenience. Alternatively, individual packages can be installed to limit dependencies._

### All

Install all dependencies.

```
pip install txtai[all]
```

### ANN

Additional ANN backends.

```
pip install txtai[ann]
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

### Console

Command line index query console.

```
pip install txtai[console]
```

### Database

Additional content storage options.

```
pip install txtai[database]
```

### Graph

Topic modeling, data connectivity and network analysis.

```
pip install txtai[graph]
```

### Model

Additional non-standard models.

```
pip install txtai[model]
```

### Pipeline

All pipelines - default install comes with most common pipelines.

```
pip install txtai[pipeline]
```

More granular extras are available for pipeline categories: `pipeline-audio`, `pipeline-data`, `pipeline-image`, `pipeline-llm`, `pipeline-text`, and `pipeline-train`.

### Scoring

Additional scoring methods.

```
pip install txtai[scoring]
```

### Vectors

Additional vector methods.

```
pip install txtai[vectors]
```

### Workflow

All workflow tasks - default install comes with most common workflow tasks.

```
pip install txtai[workflow]
```

### Combining dependencies

Multiple dependencies can be specified at the same time.

```
pip install txtai[pipeline,workflow]
```

## Environment specific prerequisites

Additional environment specific prerequisites are below.

### Linux

The AudioStream and Microphone pipelines require the [PortAudio](https://python-sounddevice.readthedocs.io/en/0.5.0/installation.html) system library. The Transcription pipeline requires the [SoundFile](https://github.com/bastibe/python-soundfile#installation) system library.

### macOS

Older versions of Faiss have a runtime dependency on `libomp` for macOS. Run `brew install libomp` in this case.

The AudioStream and Microphone pipelines require the [PortAudio](https://python-sounddevice.readthedocs.io/en/0.5.0/installation.html) system library. Run `brew install portaudio`.

### Windows

Optional dependencies require [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

The [txtai build workflow](https://github.com/neuml/txtai/blob/master/.github/workflows/build.yml) occasionally has work arounds for other known but temporary dependency issues. The [FAQ](../faq) also has a list of common problems, including common installation issues.

## CPU-only

The default install adds PyTorch with GPU support. There are a number of dependencies that come with that. When running in a CPU-only environment or using Embeddings/LLM models without PyTorch (i.e. llama.cpp or API services), the CPU-only PyTorch package can be installed with txtai as follows.

```
pip install txtai torch==[version]+cpu \
-f https://download.pytorch.org/whl/torch
```

Where `[version]` is the version of PyTorch (such as 2.4.1). The [txtai-cpu](https://hub.docker.com/r/neuml/txtai-cpu) image on Docker Hub uses this method to reduce the image size.

## Install from source

txtai can also be installed directly from GitHub to access the latest, unreleased features.

```
pip install git+https://github.com/neuml/txtai
```

Extras can be installed from GitHub by adding `#egg=txtai[<name-of-extra>]` to the end of the above URL.

## Conda

A [community-supported txtai package](https://anaconda.org/conda-forge/txtai) is available via conda-forge.

```
conda install -c conda-forge txtai
```

## Run with containers

Docker images are available for txtai. [See this section](../cloud) for more information on container-based installs.

# Installation

The easiest way to install is via pip and PyPI

    pip install txtai

You can also install txtai directly from GitHub. Using a Python Virtual Environment is recommended.

    pip install git+https://github.com/neuml/txtai

Python 3.6+ is supported.

txtai has the following optional dependencies that can be installed as extras. The patterns below are supported
in setup.py install_requires sections.

```
# Install all dependencies (mirrors txtai < 3.2)
pip install txtai[all]

# Serve txtai via a web API
pip install txtai[api]

# All pipelines - default install comes with most common pipelines
pip install txtai[pipeline]

# Word vectors, support for sentence-transformer models not on the HF Hub and additional ANN libraries
pip install txtai[similarity]

# All workflow tasks - default install comes with most common workflow tasks
pip install txtai[workflow]

# Pipelines and workflows
pip install txtai[pipeline,workflow]
```

Additional environment specific prerequisites are below.

## Linux

Optional audio transcription requires a [system library to be installed](https://github.com/bastibe/python-soundfile#installation)

## macOS

Run `brew install libomp` see [this link](https://github.com/kyamagu/faiss-wheels#prerequisite)

## Windows

Optional dependencies require [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

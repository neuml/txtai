# Set base image
ARG BASE_IMAGE=ubuntu:20.04
FROM $BASE_IMAGE

# Install GPU-enabled version of PyTorch if set
ARG GPU

# Set Python version (i.e. 3, 3.7, 3.8)
ARG PYTHON_VERSION=3

# List of txtai components to install
ARG COMPONENTS=[all]

# Locale environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN \
    # Install required packages
    apt-get update && \
    apt-get -y --no-install-recommends install libgomp1 libsndfile1 gcc g++ python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    rm -rf /var/lib/apt/lists && \
    \
    # Install txtai project and dependencies
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    python -m pip install --no-cache-dir -U pip wheel setuptools && \
    if [ -z ${GPU} ]; then pip install --no-cache-dir pip install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html; fi && \
    python -m pip install --no-cache-dir txtai${COMPONENTS} && \
    python -c "import sys, importlib.util as util; 1 if util.find_spec('nltk') else sys.exit(); import nltk; nltk.download('punkt')" && \
    \
    # Cleanup build packages
    apt-get -y purge gcc g++ python${PYTHON_VERSION}-dev && apt-get -y autoremove

# Set default working directory
WORKDIR /app

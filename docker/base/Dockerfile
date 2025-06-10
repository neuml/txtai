# Set base image
ARG BASE_IMAGE=python:3.10-slim
FROM $BASE_IMAGE

# Install GPU-enabled version of PyTorch if set
ARG GPU

# Target CPU architecture
ARG TARGETARCH

# Set Python version (i.e. 3, 3.10)
ARG PYTHON_VERSION=3

# List of txtai components to install
ARG COMPONENTS=[all]

# Locale environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN \
    # Install required packages
    apt-get update && \
    apt-get -y --no-install-recommends install libgomp1 libportaudio2 libsndfile1 gcc g++ python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    rm -rf /var/lib/apt/lists && \
    \
    # Install txtai project and dependencies
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    python -m pip install --no-cache-dir -U pip wheel setuptools && \
    if [ -z ${GPU} ] && { [ -z ${TARGETARCH} ] || [ ${TARGETARCH} = "amd64" ] ;}; then pip install --no-cache-dir torch==2.7.1+cpu torchvision==0.22.1+cpu -f https://download.pytorch.org/whl/torch -f https://download.pytorch.org/whl/torchvision; fi && \
    python -m pip install --no-cache-dir txtai${COMPONENTS} && \
    python -c "import sys, importlib.util as util; 1 if util.find_spec('nltk') else sys.exit(); import nltk; nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng'])" && \
    \
    # Cleanup build packages
    apt-get -y purge gcc g++ python${PYTHON_VERSION}-dev && apt-get -y autoremove

# Set default working directory
WORKDIR /app

# Set base image
ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

# Target CPU architecture
ARG TARGETARCH

# Set Python version (i.e. 3, 3.11)
ARG PYTHON_VERSION=3

# Locale environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN \
    # Install minimal txtai project and dependencies
    MINIMAL=1 python -m pip install --no-cache-dir txtai_minimal

# Set default working directory
WORKDIR /app

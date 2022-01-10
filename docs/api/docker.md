# Docker

A Dockerfile with commands to install txtai, all dependencies and default configuration is available in this repository.

The Dockerfile can be copied from the docker directory on GitHub locally. The following commands show how to run the API process.

```bash
docker build -t txtai.api -f docker/api.Dockerfile .
docker run --name txtai.api -p 8000:8000 --rm -it txtai.api

# Alternatively, if nvidia-docker is installed, the build will support
# GPU runtimes
docker run --name txtai.api --runtime=nvidia -p 8000:8000 --rm -it txtai.api
```

This will bring up an API instance without having to install Python, txtai or any dependencies on your machine!

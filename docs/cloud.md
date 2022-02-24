# Cloud

![cloud](images/cloud.png#only-light)
![cloud](images/cloud-dark.png#only-dark)

Scalable cloud-native applications can be built with txtai. The following runtimes are supported.

- Container Orchestration (i.e. Kubernetes)
- Docker Engine
- Serverless

Images for txtai are available on Docker Hub for [CPU](https://hub.docker.com/r/neuml/txtai-cpu) and [GPU](https://hub.docker.com/r/neuml/txtai-gpu) installs. The CPU install is recommended when GPUs aren't available given the image is half the size.

The base txtai images have no models installed and models will be downloaded each time the container starts. Caching the models is recommended as that will significantly reduce container start times. This can be done a couple different ways.

- Create a container with the [models cached](#cache-models-in-container-images)
- Set the transformers cache environment variable and mount that volume when starting the image
    ```bash
    docker run -v <local dir>:/models -e TRANSFORMERS_CACHE=/models --rm --it <docker image>
    ```

## Build txtai images

The txtai images found on Docker Hub are configured to support most situations. This image can be locally built with different options as desired.

Examples build commands below.

```bash
# Get Dockerfile
wget https://raw.githubusercontent.com/neuml/txtai/master/docker/base/Dockerfile

# Build Ubuntu 18.04 image running Python 3.7
docker build -t txtai --build-arg BASE_IMAGE=ubuntu:18.04 --build-arg PYTHON_VERSION=3.7 .

# Build image with GPU support
docker build -t txtai --build-arg GPU=1 .

# Build minimal image with the base txtai components
docker build -t txtai --build-arg COMPONENTS= .
```

## Container image model caching

As mentioned previously, model caching is recommended to reduce container start times. The following commands demonstrate this. In all cases, it is assumed a config.yml file is present in the local directory with the desired configuration set.

### API
This section builds a Docker image that caches models and starts an API service. The config.yml file should be configured with the desired components to expose via the API.

The following is a sample config.yml file that creates an Embeddings API service

```yaml
# config.yml
writable: true

embeddings:
  path: sentence-transformers/nli-mpnet-base-v2
  content: true
```

The next section builds the Docker image and starts an instance.

```bash
# Get Dockerfile
wget https://raw.githubusercontent.com/neuml/txtai/master/docker/api/Dockerfile

# CPU build
docker build -t txtai-api .

# GPU build
docker build -t txtai-api --build-arg BASE_IMAGE=neuml/txtai-gpu .

# Run
docker run -p 8000:8000 --rm -it txtai-api
```

### Service
This section builds a scheduled workflow service. [More on scheduled workflows can be found here.](../workflow/schedule)

```bash
# Get Dockerfile
wget https://raw.githubusercontent.com/neuml/txtai/master/docker/service/Dockerfile

# CPU build
docker build -t txtai-service .

# GPU build
docker build -t txtai-service --build-arg BASE_IMAGE=neuml/txtai-gpu .
```

### Workflow
This section builds a single run workflow. [Example workflows can be found here.](../examples/#workflows)

```bash
# Get Dockerfile
wget https://raw.githubusercontent.com/neuml/txtai/master/docker/workflow/Dockerfile

# CPU build
docker build -t txtai-workflow . 

# GPU build
docker build -t txtai-workflow --build-arg BASE_IMAGE=neuml/txtai-gpu .
```

## Serverless Compute

One of the most powerful features of txtai is building YAML-configured applications with the "build once, run anywhere" approach. API instances and workflows can run locally, on a server, on a cluster or serverless.

Serverless instances of txtai are supported with frameworks such as [AWS SAM](https://github.com/aws/serverless-application-model) and [Serverless](https://github.com/serverless/serverless).

The following steps shows a basic example of how to spin up a serverless API instance with AWS SAM.

- Create config.yml and template.yml

```yaml
# config.yml
writable: true

embeddings:
  path: sentence-transformers/nli-mpnet-base-v2
  content: true
```

```yaml
# template.yml
Resources:
  txtai:
    Type: AWS::Serverless::Function
    Properties:
      PackageType: Image
      MemorySize: 3000
      Timeout: 20
      Events:
        Api:
          Type: Api
          Properties:
            Path: "/{proxy+}"
            Method: ANY
    Metadata:
      Dockerfile: Dockerfile
      DockerContext: ./
      DockerTag: api
```

- Install [AWS SAM](https://pypi.org/project/aws-sam-cli/)

- Run following

```bash
# Get Dockerfile and application
wget https://raw.githubusercontent.com/neuml/txtai/master/docker/aws/api.py
wget https://raw.githubusercontent.com/neuml/txtai/master/docker/aws/Dockerfile

# Build the docker image
sam build

# Start API gateway and Lambda instance locally
sam local start-api -p 8000 --warm-containers LAZY

# Verify instance running (should return 0)
curl http://localhost:8080/count
```

If successful, a local API instance is now running in a "serverless" fashion. This configuration can be deployed to AWS using SAM. [See this link for more information.](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-cli-command-reference-sam-deploy.html)

# Generator

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Generator pipeline takes an input prompt and generates follow-on text.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Generator

# Create and run pipeline
generator = Generator()
generator("Hello, how are you?")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
generator:

# Run pipeline with workflow
workflow:
  generator:
    tasks:
      - action: generator
```

### Run with Workflows

```python
from txtai.app import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("generator", ["Hello, how are you?"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"generator", "elements": ["Hello, how are you?"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Generator.__init__
### ::: txtai.pipeline.Generator.__call__

# Sequences

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The sequences pipeline runs text through a sequence-sequence model and generates output text.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Sequences

# Create and run pipeline
sequences = Sequences()
sequences("translate English to French: Hello, how are you?")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
sequences:

# Run pipeline with workflow
workflow:
  sequences:
    tasks:
      - action: sequences
```

### Run with Workflows

```python
from txtai.app import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("sequences", ["translate English to French: Hello, how are you?"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"sequences", "elements": ["translate English to French: Hello, how are you?"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Sequences.__init__
### ::: txtai.pipeline.Sequences.__call__

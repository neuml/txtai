# URL Retrieve

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The URL Retrieve pipeline retrieves content from a HTTP(s) URL.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import URLRetrieve

# Create and run pipeline
urlretrieve = URLRetrieve()
urlretrieve("https://github.com/neuml/txtai")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
urlretrieve:

# Run pipeline with workflow
workflow:
  retrieve:
    tasks:
      - action: urlretrieve
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("urlretrieve", ["https://github.com/neuml/txtai"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"retrieve", "elements":["http://github.com/neuml/txtai"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.URLRetrieve.__init__
### ::: txtai.pipeline.URLRetrieve.__call__

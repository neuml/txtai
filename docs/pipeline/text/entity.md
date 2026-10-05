# Entity

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Entity pipeline applies a token classifier to text and extracts entity/label combinations.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Entity

# Create and run pipeline
entity = Entity()
entity("Canada's last fully intact ice shelf has suddenly collapsed, " \
       "forming a Manhattan-sized iceberg")

# Extract entities using a GLiNER model which supports dynamic labels
entity = Entity("gliner-community/gliner_medium-v2.5")
entity("Canada's last fully intact ice shelf has suddenly collapsed, " \
       "forming a Manhattan-sized iceberg", labels=["country", "city"])
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Entity extraction workflows](https://github.com/neuml/txtai/blob/master/examples/26_Entity_extraction_workflows.ipynb) | Identify entity/label combinations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/26_Entity_extraction_workflows.ipynb) |
| [Parsing the stars with txtai](https://github.com/neuml/txtai/blob/master/examples/72_Parsing_the_stars_with_txtai.ipynb) | Explore an astronomical knowledge graph of known stars, planets, galaxies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/72_Parsing_the_stars_with_txtai.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
entity:

# Run pipeline with workflow
workflow:
  entity:
    tasks:
      - action: entity
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("entity", ["Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"entity", "elements": ["Canadas last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Entity.__init__
### ::: txtai.pipeline.Entity.__call__

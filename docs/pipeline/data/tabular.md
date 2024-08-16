# Tabular

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Tabular pipeline splits tabular data into rows and columns. The tabular pipeline is most useful in creating (id, text, tag) tuples to load into Embedding indexes. 

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Tabular

# Create and run pipeline
tabular = Tabular("id", ["text"])
tabular("path to csv file")
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Transform tabular data with composable workflows](https://github.com/neuml/txtai/blob/master/examples/22_Transform_tabular_data_with_composable_workflows.ipynb) | Transform, index and search tabular data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/22_Transform_tabular_data_with_composable_workflows.ipynb) |

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
tabular:
    idcolumn: id
    textcolumns:
      - text

# Run pipeline with workflow
workflow:
  tabular:
    tasks:
      - action: tabular
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("tabular", ["path to csv file"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"tabular", "elements":["path to csv file"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Tabular.__init__
### ::: txtai.pipeline.Tabular.__call__

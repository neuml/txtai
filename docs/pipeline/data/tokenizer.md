# Tokenizer

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The Tokenizer pipeline splits text into tokens. This is primarily used for keyword / term indexing.

_Note: Transformers-based models have their own tokenizers and this pipeline isn't designed for working with Transformers models._

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import Tokenizer

# Create and run pipeline
tokenizer = Tokenizer()
tokenizer("text to tokenize")

# Whitespace tokenization
tokenizer = Tokenizer(whitespace=True)
tokenizer("text to tokenize")

# Tokenize using a regular expression
tokenizer = Tokenizer(regexp=r"\w{5,}")
tokenizer("text to tokenize")

# Tokenize into trigrams like pg_trgm
tokenizer = Tokenizer(ngrams={
  "ngrams": 3, "lpad": "  ", "rpad": " ", "unique": True
})
tokenize("text to tokenize")

# Tokenize into edge ngrams
tokenizer = Tokenizer(ngrams={"nmin": 2, "nmax": 5, "edge": True})
tokenizer("text to tokenize")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
tokenizer:

# Run pipeline with workflow
workflow:
  tokenizer:
    tasks:
      - action: tokenizer
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("tokenizer", ["text to tokenize"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"tokenizer", "elements":["text"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.Tokenizer.__init__
### ::: txtai.pipeline.Tokenizer.__call__

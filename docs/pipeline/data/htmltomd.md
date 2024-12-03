# HTML To Markdown 

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The HTML To Markdown pipeline transforms HTML to Markdown.

Markdown formatting is applied for headings, blockquotes, lists, code, tables and text. Visual formatting is also included (bold, italic etc).

This pipeline searches for the best node that has relevant text, often found with an `article`, `main` or `body` tag.

The HTML to Markdown pipeline requires the [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) library to be installed.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import HTMLToMarkdown

# Create and run pipeline
md = HTMLToMarkdown()
md("<html><body>This is a test</body></html>")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
htmltomarkdown:

# Run pipeline with workflow
workflow:
  markdown:
    tasks:
      - action: htmltomarkdown
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("markdown", ["<html><body>This is a test</body></html>"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"markdown", "elements":["<html><body>This is a test</body></html>"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.HTMLToMarkdown.__init__
### ::: txtai.pipeline.HTMLToMarkdown.__call__

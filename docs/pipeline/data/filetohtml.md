# File To HTML

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

The File To HTML pipeline transforms files to HTML. It supports the following text extraction backends.

## Apache Tika

[Apache Tika](https://tika.apache.org/) detects and extracts metadata and text from over a thousand different file types. See [this link](https://tika.apache.org/2.9.2/formats.html) for a list of supported document formats.

Apache Tika requires [Java](https://en.wikipedia.org/wiki/Java_(programming_language)) to be installed. An alternative to that is starting a separate Apache Tika service via [this Docker Image](https://hub.docker.com/r/apache/tika) and setting these [environment variables](https://github.com/chrismattmann/tika-python?tab=readme-ov-file#environment-variables).

## Docling

[Docling](https://github.com/DS4SD/docling) parses documents and exports them to the desired format with ease and speed. This is a library that has rapidly gained popularity starting in late 2024. Docling excels in parsing formatting elements from PDFs (tables, sections etc).

See [this link](https://github.com/DS4SD/docling?tab=readme-ov-file#features) for a list of supported document formats.

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import FileToHTML

# Create and run pipeline
html = FileToHTML()
html("/path/to/file")
```

## Configuration-driven example

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../../../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../../../workflow/#configuration-driven-example) or the [API](../../../api#local-instance).

### config.yml
```yaml
# Create pipeline using lower case class name
filetohtml:

# Run pipeline with workflow
workflow:
  html:
    tasks:
      - action: filetohtml
```

### Run with Workflows

```python
from txtai import Application

# Create and run pipeline with workflow
app = Application("config.yml")
list(app.workflow("html", ["/path/to/file"]))
```

### Run with API

```bash
CONFIG=config.yml uvicorn "txtai.api:app" &

curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -d '{"name":"html", "elements":["/path/to/file"]}'
```

## Methods

Python documentation for the pipeline.

### ::: txtai.pipeline.FileToHTML.__init__
### ::: txtai.pipeline.FileToHTML.__call__

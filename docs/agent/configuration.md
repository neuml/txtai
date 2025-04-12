# Configuration

An agent takes two main arguments, an LLM and a list of tools.

The txtai agent framework is built with [smolagents](https://github.com/huggingface/smolagents). Additional options can be passed in the `Agent` constructor.

```python
from datetime import datetime

from txtai import Agent

wikipedia = {
    "name": "wikipedia",
    "description": "Searches a Wikipedia database",
    "provider": "huggingface-hub",
    "container": "neuml/txtai-wikipedia"
}

arxiv = {
    "name": "arxiv",
    "description": "Searches a database of scientific papers",
    "provider": "huggingface-hub",
    "container": "neuml/txtai-arxiv"
}

def today() -> str:
    """
    Gets the current date and time

    Returns:
        current date and time
    """

    return datetime.today().isoformat()

agent = Agent(
    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    tools=[today, wikipedia, arxiv, "websearch"],
)
```

## model

```yaml
model: string|llm instance
```

LLM model path or LLM pipeline instance. The `llm` parameter is also supported for backwards compatibility.

See the [LLM pipeline](../../pipeline/text/llm) for more information.

## tools

```yaml
tools: list
```

List of tools to supply to the agent. Supports the following configurations.

### function

A function tool takes the following dictionary fields.

| Field       | Description              |
|:------------|:-------------------------|
| name        | name of the tool         |
| description | tool description         |
| target      | target method / callable |

A function or callable method can also be directly supplied in the `tools` list. In this case, the fields are inferred from the method documentation.

### embeddings

Embeddings indexes have built-in support. Provide the following dictionary configuration to add an embeddings index as a tool.

| Field       | Description                                |
|:------------|:-------------------------------------------|
| name        | embeddings index name                      |
| description | embeddings index description               | 
| **kwargs    | Parameters to pass to [embeddings.load](../../embeddings/methods/#txtai.embeddings.Embeddings.load) |

### tool

A tool instance can be provided. Additionally, the following strings load tools directly.

| Tool        | Description                                               |
|:------------|:----------------------------------------------------------|
| http.*      | HTTP Path to a Model Context Protocol (MCP) server        |
| python      | Runs a Python action                                      |
| websearch   | Runs a websearch using the built-in websearch tool        |
| webview     | Extracts content from a web page                          |

## method

```yaml
method: code|tool
```

Sets the agent method. Supports either a `code` or `tool` calling agent (default). A code agent generates Python code and executes that. A tool calling agent generates JSON blocks and calls the agents within those blocks.

[Read more on this here](https://huggingface.co/docs/smolagents/v1.13.0/en/guided_tour#codeagent-and-toolcallingagent).

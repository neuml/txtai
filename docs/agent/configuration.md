# Configuration

An agent takes two main arguments, an LLM and a list of tools.

The txtai agent framework is built with [Transformers Agents](https://huggingface.co/docs/transformers/en/agents) and additional options can be directly passed in the `Agent` constructor.

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
    tools=[today, wikipedia, arxiv, "websearch"],
    llm="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
)
```

## llm

```yaml
llm: string|llm instance
```

LLM path or LLM pipeline instance. See the [LLM pipeline](../../pipeline/text/llm) for more information.

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

### transformers

A Transformers tool instance can be provided. Additionally, the following strings load tools directly from Transformers.

| Tool        | Description                                               |
|:------------|:----------------------------------------------------------|
| websearch   | Runs a websearch using built-in Transformers Agent tool   |

## method

```yaml
method: reactjson|reactcode|code
```

Sets the agent method. Defaults to `reactjson`. [Read more on this here](https://huggingface.co/docs/transformers/en/agents#types-of-agents).

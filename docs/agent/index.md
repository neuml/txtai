# Agent

![agent](../images/agent.png)

An agent automatically creates workflows to answer multi-faceted user requests. Agents iteratively prompt and/or interface with tools to
step through a process and ultimately come to an answer for a request.

Agents excel at complex tasks where multiple tools and/or methods are required. They incorporate a level of randomness similar to different
people working on the same task. When the request is simple and/or there is a rule-based process, other methods such as RAG and Workflows
should be explored.

The following code snippet shows a basic agent.

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
    max_iterations=10,
)
```

The agent above has access to a two embeddings database (Wikipedia and ArXiv) and the web. Given the user's input request, the Agent decides the best tool to solve the task. In this case, a web search is executed.

## Example

The first example will solve a problem with multiple data points. See below.

```python
agent("Which city has the highest population, Boston or New York?")
```

This requires looking up the populations of each city before knowing how to answer the question.

## Agentic RAG

Standard Retrieval Augmented Generation (RAG) runs a single vector search query to obtain a context and builds a prompt with the context + input question. Agentic RAG is a more complex process that goes through multiple iterations. It can also utilize multiple databases to come to a final conclusion.

The example below aggregates information from multiple examples and builds a report on a topic.

```python
researcher = """
You're an expert researcher looking to write a paper on {topic}.
Search for websites, scientific papers and Wikipedia related to the topic.
Write a report with summaries and references (with hyperlinks).
Write the text as Markdown.
"""

agent(researcher.format(topic="alien life"))
```

## Agent Teams

Agents can also be tools. This enables the concept of building "Agent Teams" to solve problems. The previous example can be rewritten as a list of agents.

```python
from txtai import Agent, LLM

llm = LLM("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

websearcher = Agent(
    tools=["websearch"],
    llm=llm,
)

wikiman = Agent(
    tools=[{
        "name": "wikipedia",
        "description": "Searches a Wikipedia database",
        "provider": "huggingface-hub",
        "container": "neuml/txtai-wikipedia"
    }],
    llm=llm,
)

researcher = Agent(
    tools=[{
        "name": "arxiv",
        "description": "Searches a database of scientific papers",
        "provider": "huggingface-hub",
        "container": "neuml/txtai-arxiv"
    }],
    llm=llm,
)

agent = Agent(
    tools=[{
        "name": "websearcher",
        "description": "I run web searches, there is no answer a web search can't solve!",
        "target": websearcher
    }, {
        "name": "wikiman",
        "description": "Wikipedia has all the answers, I search Wikipedia and answer questions",
        "target": wikiman
    }, {
        "name": "researcher",
        "description": "I'm a science guy. I search arXiv to get all my answers.",
        "target": researcher
    }],
    llm=llm,
    max_iterations=10
)
```

This provides another level of intelligence to the process. Instead of just a single tool execution, each agent-tool combination has it's own reasoning engine.

```python
agent("""
Work with your team and build a comprehensive report on fundamental
concepts about Signal Processing.
Write the output in Markdown.
""")
```

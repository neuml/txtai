"""
Agent Quick Start
Easy to use way to get started with AI Agents.

TxtAI has many example notebooks covering everything the framework provides
Examples: https://neuml.github.io/txtai/examples

Install TxtAI
  pip install txtai[agent]
"""

# pylint: disable=C0103
from datetime import datetime
from txtai import Agent

# Step 1: Define your Embeddings database
#
# Replace provider/container with a path to a local Embeddings database
# See RAG Quickstart for an example of building your own custom database
embeddings = {
    "name": "wikipedia",
    "description": "Searches a Wikipedia database",
    # "path": "path to your embeddings database"
    "provider": "huggingface-hub",
    "container": "neuml/txtai-wikipedia",
}


# Step 2: Define other tools
#
# Add any Python function. Just need to describe it.
def today() -> str:
    """
    Gets the current date and time

    Returns:
        current date and time
    """

    return datetime.today().isoformat()


# Step 3: Create a list of available tools
#
# Combine defined tools with default tools
tools = [
    embeddings,  # Embeddings database with YOUR data
    today,  # Python function
    "websearch",  # Runs a websearch using default engine
    "webview",  # Loads a web page
]

# Step 4: Set LLM configuration
#
# LLM APIs
#  model = "gpt-5.1"
#  model = "claude-opus-4-5-20251101"
#  model = "gemini/gemini-3-pro-preview"
#
# Local LLMs
#  model = "ollama/gpt-oss
#  model = "openai/gpt-oss-20b"
#  model = "unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf"
#
# Pass multiple options as a dictionary
#  model = {
#    "path": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
#    "n_ctx": 25000
#  }
model = "Qwen/Qwen3-4B-Instruct-2507"

# Step 4: Create an Agent
#
# Set LLM, tools and other configuration
# See this for more options: https://huggingface.co/docs/smolagents/reference/agents#agents
agent = Agent(model=model, tools=tools, max_steps=10)

print(agent("Tell me about the Roman Empire"))
print(agent("What is the current date?"))
print(agent("Get the 5 top news stories for today", maxlength=25000))

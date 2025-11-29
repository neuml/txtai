"""
RAG Quick Start
Easy to use way to get started with RAG using YOUR data

For a complete application see this: https://github.com/neuml/rag

TxtAI has many example notebooks covering everything the framework provides
Examples: https://neuml.github.io/txtai/examples

Install TxtAI
  pip install txtai[pipeline-data]
"""

# pylint: disable=C0103
import os

from txtai import Embeddings, RAG
from txtai.pipeline import Textractor

# Step 1: Collect files from local directory
#
# Defaults to "data". Set to whereever your files are.
path = "data"
files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Step 2: Text Extraction / Chunking
#
# Using section based chunking here. More complex options available such as semantic chunking, iterative chunking etc.
# Documentation: https://neuml.github.io/txtai/pipeline/data/textractor
# Supports Chonkie chunking as well: https://docs.chonkie.ai/oss/chunkers/overview
textractor = Textractor(backend="docling", sections=True)
chunks = []
for f in files:
    for chunk in textractor(f):
        chunks.append((f, chunk))

# Step 3: Build an embeddings database
#
# The `path` parameter sets the vector embeddings model. Supports Hugging Face models, llama.cpp, Ollama, vLLM and more.
# Documentation: https://neuml.github.io/txtai/embeddings/
embeddings = Embeddings(content=True, path="Qwen/Qwen3-Embedding-0.6B", maxlength=2048)
embeddings.index(chunks)

# Step 4: Create RAG pipeline
#
# Combines an embeddings database and an LLM.
# Supports Hugging Face models, llama.cpp, Ollama, vLLM and more
# Documentation: https://neuml.github.io/txtai/pipeline/text/rag

# User prompt template
template = """
  Answer the following question using the provided context.

  Question:
  {question}

  Context:
  {context}
"""

rag = RAG(
    embeddings,
    "Qwen/Qwen3-0.6B",
    system="You are a friendly assistant",
    template=template,
    output="flatten",
)

question = "Summarize the main advancements made by BERT"
print(rag(question, maxlength=2048, stripthink=True))

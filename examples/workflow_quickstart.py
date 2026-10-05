"""
Workflow Quick Start
Easy to use way to get started with deterministic workflows.

TxtAI has many example notebooks covering everything the framework provides
Examples: https://neuml.github.io/txtai/examples

Install TxtAI
  pip install txtai[pipeline-data]
"""

from txtai import LLM, Workflow
from txtai.pipeline import Summary, Textractor, Translation
from txtai.workflow import Task

# Step 1: Define available pipelines
textractor = Textractor(backend="docling", headers={"user-agent": "Mozilla/5.0"})
summary = Summary()
translate = Translation()

# Step 2: Define workflow tasks
workflow = Workflow([Task(textractor), Task(summary), Task(lambda inputs: [translate(x, "fr") for x in inputs])])

# Step 3: Run the workflow
print(list(workflow(["https://neuml.com"])))

# Each component above is a single model that specializes in a task
# LLMs can also be used to accomplish the same tasks


# pylint: disable=E0102,C0116
def summary(text):
    return f"""
Summarize the following text in 40 words or less.

{text}
"""


def translate(text, language):
    return f"""
Translate the following text to {language}.

{text}
"""


textractor = Textractor(backend="docling", headers={"user-agent": "Mozilla/5.0"})
llm = LLM("Qwen/Qwen3-4B-Instruct-2507")

workflow = Workflow(
    [
        Task(textractor),
        Task(lambda inputs: llm([summary(x) for x in inputs], maxlength=25000, defaultrole="user")),
        Task(lambda inputs: llm([translate(x, "fr") for x in inputs], maxlength=25000, defaultrole="user")),
    ]
)

print(list(workflow(["https://neuml.com"])))

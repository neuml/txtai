# Workflows

Workflows are a simple yet powerful construct that takes a callable and returns elements. Workflows don't know they are working with pipelines but enable efficient processing of pipeline data. Workflows are streaming by nature and work on data in batches, allowing large volumes of data to be processed efficiently.

An example Workflow is shown below. This workflow will work with both documents and audio files. Documents will have text extracted and summarized. Audio files will be transcribed. Both results will be joined, translated into French and loaded into an Embeddings index.

```python
# file:// prefixes are required to signal to the workflow this is a file and not a text string
files = [
    "file://txtai/article.pdf",
    "file://txtai/US_tops_5_million.wav",
    "file://txtai/Canadas_last_fully.wav",
    "file://txtai/Beijing_mobilises.wav",
    "file://txtai/The_National_Park.wav",
    "file://txtai/Maine_man_wins_1_mil.wav",
    "file://txtai/Make_huge_profits.wav"
]

data = [(x, element, None) for x, element in enumerate(files)]

# Workflow that extracts text and builds a summary
articles = Workflow([
    FileTask(textractor),
    Task(lambda x: summary([y[:1024] for y in x]))
])

# Define workflow tasks. Workflows can also be tasks!
tasks = [
    WorkflowTask(articles, r".\.pdf$"),
    FileTask(transcribe, r"\.wav$"),
    Task(lambda x: translate(x, "fr")),
    Task(index, unpack=False)
]

# Workflow that translates text to French
workflow = Workflow(tasks)
for _ in workflow(data):
    pass
```

::: txtai.workflow.Workflow.__init__
::: txtai.workflow.Workflow.__call__

## Tasks

::: txtai.workflow.Task.__init__

### File Task

Task that processes file urls

### Image Task

Task that processes image urls

### Service Task

Task that runs content against a http service

### Storage Task

Task that expands a local directory or cloud storage bucket into a list of URLs to process

### URL Task

Task that processes urls

### Workflow Task

Task that runs a Workflow. Allows creating workflows of workflows.

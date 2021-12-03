# Workflow

Workflows are a simple yet powerful construct that takes a callable and returns elements. Workflows operate well with pipelines but can work with any callable object. Workflows are streaming by nature and work on data in batches, allowing large volumes of data to be processed efficiently.

Given that pipelines are callable objects, workflows enable efficient processing of pipeline data. Transformers models typically work with smaller batches of data, workflows are well suited to feed a series of transformers pipelines. 

An example of the most basic workflow:

```python
workflow = Workflow([Task(lambda x: [y * 2 for y in x])])
list(workflow([1, 2, 3]))
```

This example simply multiplies each input value and returns a outputs via a generator. 

A more complex example is shown below. This workflow will work with both documents and audio files. Documents will have text extracted and summarized. Audio files will be transcribed. Both results will be joined, translated into French and loaded into an Embeddings index.

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

Workflows can be defined using Python as described here but they can also be created as YAML configuration and run as shown below.

```python
# Create and run the workflow
app = API("workflow.yml")
data = list(app.workflow("workflow", ["input text"]))
```

[Read more here on creating workflow YAML](../../api). 

# Workflow

Workflows are callable objects. Workflows take an input of iterable data elements and output iterable data elements. 

### ::: txtai.workflow.Workflow.__init__
### ::: txtai.workflow.Workflow.__call__

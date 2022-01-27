# Retrieve Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Retrieve Task connects to a url and downloads the content locally. This task is helpful when working with actions that require data to be available locally.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import RetrieveTask, Workflow

workflow = Workflow([RetrieveTask(directory="/tmp")])
workflow(["https://file.to.download", "/local/file/to/copy"])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: retrieve
      directory: /tmp
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.RetrieveTask.__init__
### ::: txtai.workflow.RetrieveTask.register

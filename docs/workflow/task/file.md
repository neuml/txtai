# File Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The File Task validates a file exists. It handles both file paths and local file urls. Note that this task _only_ works with local files.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import FileTask, Workflow

workflow = Workflow([FileTask()])
workflow(["/path/to/file", "file:///path/to/file"])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: file
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.FileTask.__init__

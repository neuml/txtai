# Url Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Url Task validates that inputs start with a url prefix.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import UrlTask, Workflow

workflow = Workflow([UrlTask()])
workflow(["https://file.to.download", "file:////local/file/to/copy"])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: url
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.UrlTask.__init__

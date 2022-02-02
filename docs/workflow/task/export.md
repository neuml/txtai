# Export Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Export Task exports task outputs to CSV or Excel.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import FileTask, Workflow

workflow = Workflow([ExportTask()])
workflow(["Input 1", "Input2"])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: export
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.ExportTask.__init__
### ::: txtai.workflow.ExportTask.register

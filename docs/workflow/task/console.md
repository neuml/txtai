# Console Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Console Task prints task inputs and outputs to standard output. This task is mainly used for debugging and can be added at any point in a workflow.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import FileTask, Workflow

workflow = Workflow([ConsoleTask()])
workflow(["Input 1", "Input2"])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: console
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.ConsoleTask.__init__

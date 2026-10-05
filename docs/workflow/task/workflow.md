# Workflow Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Workflow Task runs a workflow. Allows creating workflows of workflows.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import WorkflowTask, Workflow

workflow = Workflow([WorkflowTask(otherworkflow)])
workflow(["input data"])
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.WorkflowTask.__init__

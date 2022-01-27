# Service Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Service Task extracts content from a http service.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import ServiceTask, Workflow

workflow = Workflow([ServiceTask(url="https://service.url/action)])
workflow(["parameter"])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: service
      url: https://service.url/action
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.ServiceTask.__init__
### ::: txtai.workflow.ServiceTask.register

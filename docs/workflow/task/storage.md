# Storage Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Storage Task expands a local directory or cloud storage bucket into a list of URLs to process.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import StorageTask, Workflow

workflow = Workflow([StorageTask()])
workflow(["s3://path/to/bucket", "local://local/directory"])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: storage
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.StorageTask.__init__

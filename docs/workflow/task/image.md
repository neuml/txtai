# Image Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Image Task reads file paths, check the file is an image and opens it as an Image object. Note that this task _only_ works with local files.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import ImageTask, Workflow

workflow = Workflow([ImageTask()])
workflow(["image.jpg", "image.gif"])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: image
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.ImageTask.__init__

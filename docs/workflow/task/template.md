# Template Task

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

The Template Task generates text from a template and task inputs. Templates can be used to prepare data for a number of tasks including generating large
language model (LLM) prompts.

## Example

The following shows a simple example using this task as part of a workflow.

```python
from txtai.workflow import TemplateTask, Workflow

workflow = Workflow([TemplateTask(template="This is a {text} task")])
workflow([{"text": "template"}])
```

## Configuration-driven example

This task can also be created with workflow configuration.

```yaml
workflow:
  tasks:
    - task: template
      template: This is a {text} task
```

## Methods

Python documentation for the task.

### ::: txtai.workflow.TemplateTask.__init__

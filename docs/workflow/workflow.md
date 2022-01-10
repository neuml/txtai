# Workflow Task

Task that runs a Workflow. Allows creating workflows of workflows.

```python
workflow = Workflow([WorkflowTask(workflow)])
workflow(["input data"])
```

::: txtai.workflow.WorkflowTask.__init__

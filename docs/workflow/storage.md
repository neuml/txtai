## Storage Task

Task that expands a local directory or cloud storage bucket into a list of URLs to process.

```python
workflow = Workflow([StorageTask()])
workflow(["s3://path/to/bucket", "local://local/directory"])
```

::: txtai.workflow.StorageTask.__init__

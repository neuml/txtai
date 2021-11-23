## Retrieve Task

Task that connects to a url and downloads the content locally.

```python
workflow = Workflow([RetrieveTask(directory="/tmp")])
workflow(["https://file.to.download", "/local/file/to/copy"])
```

::: txtai.workflow.RetrieveTask.__init__

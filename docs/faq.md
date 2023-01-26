# FAQ

Below is a list of frequently asked questions and common issues encountered.

----------

__Issue__

Embeddings query errors like this:

```
SQLError: no such function: json_extract
```

__Solution__

Upgrade Python version as it doesn't have SQLite support for json_extract

----------

__Issue__

Segmentation faults and similar errors on macOS

__Solution__

Downgrade PyTorch to <= 1.12. See issue [#377](https://github.com/neuml/txtai/issues/377) for more on this issue. 

----------

__Issue__

`ContextualVersionConflict` exception when importing certain libraries while running one of the [examples](./examples.md) notebooks on Google Colab

__Solution__

Restart the kernel. See issue [#409](https://github.com/neuml/txtai/issues/409) for more on this issue. 

----------
# Why txtai?

![why](images/why.png#only-light)
![why](images/why-dark.png#only-dark)

In addition to traditional search systems, a growing number of semantic search solutions are available, so why txtai?

- `pip install txtai` is all you need
```python
# Get started in a couple lines
from txtai.embeddings import Embeddings

embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
embeddings.index([(0, "Correct", None), (1, "Not what we hoped", None)])
embeddings.search("positive", 1)
#[(0, 0.2986203730106354)]
```
- Works well with both small and big data - scale up as needed
- Rich data processing framework (pipelines and workflows) to pre and post process data
- Work in your programming language of choice via the API
- Modular with low footprint - install additional dependencies when you need them
- [Learn by example](../examples) - notebooks cover all available functionality

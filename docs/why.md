# Why txtai?

![why](images/why.png#only-light)
![why](images/why-dark.png#only-dark)

In addition to traditional search systems, a growing number of language model backed solutions are available, so why txtai?

- Up and running in minutes with [pip](../install/) or [Docker](../cloud/)
```python
# Get started in a couple lines
from txtai.embeddings import Embeddings

embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
embeddings.index([(0, "Correct", None), (1, "Not what we hoped", None)])
embeddings.search("positive", 1)
#[(0, 0.2986203730106354)]
```
- Built-in API makes it easy to develop applications using your programming language of choice
```yaml
# app.yml
embeddings:
    path: sentence-transformers/all-MiniLM-L6-v2
```
```bash
CONFIG=app.yml uvicorn "txtai.api:app"
curl -X GET "http://localhost:8000/search?query=positive"
```
- Run local - no need to ship data off to disparate remote services
- Work with micromodels all the way up to large language models (LLMs)
- Low footprint - install additional dependencies and scale up when needed
- [Learn by example](../examples) - notebooks cover all available functionality

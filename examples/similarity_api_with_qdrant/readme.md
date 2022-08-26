# Running txtai with Qdrant backend

Qdrant - is fast and scalable vector search engine, which CRUD API along with similarity search API.

Qdrant can replace FAISS/Annoy/HNSWLIB indexes in production environment, which requires high availability, high throughput and flexibility.


## Run Qdrant

```
docker run -p 6333:6333 qdrant/qdrant
```

More info on [Qdrant](https://github.com/qdrant/qdrant).

## Configuration

```yaml
# Embeddings index
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  backend: qdrant
  qdrant:
    collection_name: Example  # Collection name used in Qdrant
    connection:
      host: localhost # Host of the qdrant service
    collection:
      hnsw_config:
        ef_construct: 150
```

With this configuration, Qdrant will be used as a backend for embeddings index.

To run example:
```
CONFIG=examples/similarity_api_with_qdrant/config.yml uvicorn "txtai.api:app"
```
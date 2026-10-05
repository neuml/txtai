# Distributed embeddings clusters

The API supports combining multiple API instances into a single logical embeddings index. An example configuration is shown below.

```yaml
cluster:
    shards:
        - http://127.0.0.1:8002
        - http://127.0.0.1:8003
```

This configuration aggregates the API instances above as index shards. Data is evenly split among each of the shards at index time. Queries are run in parallel against each shard and the results are joined together. This method allows horizontal scaling and supports very large index clusters.

This method is only recommended for data sets in the 1 billion+ records. The ANN libraries can easily support smaller data sizes and this method is not worth the additional complexity. At this time, new shards can not be added after building the initial index.

See the link below for a detailed example covering distributed embeddings clusters.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Distributed embeddings cluster](https://github.com/neuml/txtai/blob/master/examples/15_Distributed_embeddings_cluster.ipynb) | Distribute an embeddings index across multiple data nodes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/15_Distributed_embeddings_cluster.ipynb) |

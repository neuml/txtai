# Graph

Enable graph storage via the `graph` parameter. This component requires the [graph](../../../install/#graph) extras package.

When enabled, a graph network is built using the embeddings index. Graph nodes are synced with each embeddings index operation (index/upsert/delete). Graph edges are created using the embeddings index upon completion of each index/upsert/delete embeddings index call.

## backend
```yaml
backend: networkx|rdbms|custom
```

Sets the graph backend. Defaults to `networkx`.

Add custom graph storage engines via setting this parameter to the fully resolvable class string.

The `rdbms` backend has the following additional settings.

### rdbms
```yaml
url: database url connection string, alternatively can be set via the
     GRAPH_URL environment variable
schema: database schema to store graph - defaults to being
        determined by the database
nodes: table to store node data, defaults to `nodes`
edges: table to store edge data, defaults to `edges`
```

## batchsize
```yaml
batchsize: int
```

Batch query size, used to query embeddings index - defaults to 256.

## limit
```yaml
limit: int
```

Maximum number of results to return per embeddings query - defaults to 15.

## minscore
```yaml
minscore: float
```

Minimum score required to consider embeddings query matches - defaults to 0.1.

## approximate
```yaml
approximate: boolean
```

When true, queries only run for nodes without edges - defaults to true.

## topics
```yaml
topics:
    algorithm: community detection algorithm (string), options are
               louvain (default), greedy, lpa
    level: controls number of topics (string), options are best (default) or first
    resolution: controls number of topics (int), larger values create more
                topics (int), defaults to 100
    labels: scoring index method used to build topic labels (string)
            options are bm25 (default), tfidf, sif
    terms: number of frequent terms to use for topic labels (int), defaults to 4
    stopwords: optional list of stop words to exclude from topic labels
    categories: optional list of categories used to group topics, allows
                granular topics with broad categories grouping topics
```

Enables topic modeling. Defaults are tuned so that in most cases these values don't need to be changed (except for categories). These parameters are available for advanced use cases where one wants full control over the community detection process.

## copyattributes
```yaml
copyattributes: boolean|list
```

Copy these attributes from input dictionaries in the `insert` method. If this is set to `True`, all attributes are copied. Otherwise, only the
attributes specified in this list are copied to the graph as attributes.

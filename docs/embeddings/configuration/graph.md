# Graph

The following covers available graph configuration options.

## graph
```yaml
graph:
    backend: graph network backend (string), defaults to "networkx"
    batchsize: batch query size, used to query embeddings index (int)
               defaults to 256
    limit: maximum number of results to return per embeddings query (int)
           defaults to 15
    minscore: minimum score required to consider embeddings query matches (float)
              defaults to 0.1
    approximate: when true, queries only run for nodes without edges (boolean)
                 defaults to true
    topics: see below
```

Enables graph storage. When set, a graph network is built using the embeddings index. Graph nodes are synced with each embeddings index operation (index/upsert/delete). Graph edges are created using the embeddings index upon completion of each index/upsert/delete embeddings index call.

Add custom graph storage engines via setting the `graph.backend` parameter to the fully resolvable class string.

Defaults are tuned so that in most cases these values don't need to be changed. 

### topics
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

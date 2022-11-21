# Index guide

![indexing](../images/indexing.png#only-light)
![indexing](../images/indexing-dark.png#only-dark)

This section gives an in-depth overview on how to index data with txtai. We'll cover vectorization, indexing, updating and deleting data.

## Vectorization

The most compute intensive step in building an index is vectorization. The [path](../configuration#path) parameter sets the path to the vector model. There is logic to automatically detect the vector model [method](../configuration#method) but it can also be set directly.

The [batch](../configuration#batch) and [encodebatch](../configuration#encodebatch) parameters control the vectorization process. Larger values for `batch` will pass larger batches to the vectorization method. Larger values for `encodebatch` will pass larger batches for each vector encode call. In the case of GPU vector models, larger values will consume more GPU memory.

Data is buffered to temporary storage during indexing as embeddings vectors can be quite large (for example 768 dimensions of float32 is 768 * 4 = 3072 bytes per vector). Once vectorization is complete, a mmapped array is created with all vectors for [Approximate Nearest Neighbor (ANN)](../configuration#backend) indexing.

## Setting a backend

As mentioned above, computed vectors are stored in an ANN. There are various index [backends](../configuration#backend) that can be configured. Faiss is the default backend.

## Content storage

Embeddings indexes can optionally [store content](../configuration#content). When this is enabled, the input content is saved in a database alongside the computed vectors. This enables filtering on additional fields and content retrieval.

## Index vs Upsert

Data is loaded into an index with either an [index](../methods#txtai.embeddings.base.Embeddings.index) or [upsert](../methods#txtai.embeddings.base.Embeddings.upsert) call.

```python
embeddings.index([(uid, text, None) for uid, text in enumerate(data)])
embeddings.upsert([(uid, text, None) for uid, text in enumerate(data)])
```

The `index` call will build a brand new index replacing an existing one. `upsert` will insert or update records. `upsert` ops do _not_ require a full index rebuild.

## Save

Indexes can be stored in a directory using the [save](../methods/#txtai.embeddings.base.Embeddings.save) method.

```python
embeddings.save("/path/to/save")
```

Compressed indexes are also supported.

```python
embeddings.save("/path/to/save/index.tar.gz")
```

In addition to saving indexes locally, they can also be persisted to [cloud storage](../configuration/cloud).

```python
embeddings.save("/path/to/save/index.tar.gz", cloud={...})
```

This is especially useful when running in a serverless context or otherwise running on temporary compute. Cloud storage is only supported with compressed indexes.

Embeddings indexes can be restored using the [load](../methods/#txtai.embeddings.base.Embeddings.load) method.

```python
embeddings.load("/path/to/load")
```

## Delete

Content can be removed from the index with the [delete](../methods#txtai.embeddings.base.Embeddings.delete) method. This method takes a list of ids to delete.

```python
embeddings.delete(ids)
```

## Reindex

When [content storage](../configuration#content) is enabled, [reindex](../methods#txtai.embeddings.base.Embeddings.reindex) can be called to rebuild the index with new settings. For example, the backend can be switched from faiss to hnsw or the vector model can be updated. This prevents having to go back to the original raw data. 

```python
embeddings.reindex({"path": "sentence-transformers/all-MiniLM-L6-v2", "backend": "hnsw"})
```

## Graph

Dimensionality reduction with UMAP combined with HDBSCAN is a popular topic modeling method found in a number of libraries. txtai takes a different approach with a semantic graph.

Enabling a [graph network](../configuration#graph) adds a semantic graph at index time as data is being vectorized. Vector embeddings are used to create relationships in the graph. Finally, community detection algorithms build topic clusters. Semantic graphs can also be used to analyze data connectivity.

This approach has the advantage of only having to vectorize data once. It also has the advantage of better topic precision given there isn't a dimensionality reduction operation (UMAP). Semantic graph examples are shown below.

Get a mapping of discovered topics to associated ids.

```python
embeddings.graph.topics
```

Show the most central nodes in the index.

```python
embeddings.graph.centrality()
```

Show how node 1 and node 2 are connected in the graph.

```python
embeddings.graph.showpath(id1, id2)
```

Graphs are persisted alongside an embeddings index. Each save and load will also save and load the graph.

## Scoring

When using [word vector backed models](../configuration#words) with scoring set, a separate call is required before calling `index` as follows:

```python
embeddings.score([(uid, text, None) for uid, text in enumerate(data)])
embeddings.index([(uid, text, None) for uid, text in enumerate(data)])
```

Two calls are required to support generator-backed iteration of data. The scoring index requires a separate full-pass of the data.

Scoring instances can also create a standalone keyword-based index (BM25, TF-IDF). See [this link](../../examples/#semantic-search) to learn more.

# Embeddings

![embeddings](../images/embeddings.png#only-light)
![embeddings](../images/embeddings-dark.png#only-dark)

Embeddings is the engine that delivers semantic search. Data is transformed into embeddings vectors where similar concepts will produce similar vectors. Indexes both large and small are built with these vectors. The indexes are used to find results that have the same meaning, not necessarily the same keywords.

The following code snippet shows how to build and search an embeddings index.

```python
from txtai.embeddings import Embeddings

# Create embeddings model, backed by sentence-transformers & transformers
embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})

data = [
  "US tops 5 million confirmed virus cases",
  "Canada's last fully intact ice shelf has suddenly collapsed, " +
  "forming a Manhattan-sized iceberg",
  "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
  "The National Park Service warns against sacrificing slower friends " +
  "in a bear attack",
  "Maine man wins $1M from $25 lottery ticket",
  "Make huge profits without work, earn up to $100,000 a day"
]

# Create an index for the list of text
embeddings.index([(uid, text, None) for uid, text in enumerate(data)])

print("%-20s %s" % ("Query", "Best Match"))
print("-" * 50)

# Run an embeddings search for each query
for query in ("feel good story", "climate change", "public health story", "war",
              "wildlife", "asia", "lucky", "dishonest junk"):
    # Extract uid of first result
    # search result format: (uid, score)
    uid = embeddings.search(query, 1)[0][0]

    # Print text
    print("%-20s %s" % (query, data[uid]))
```

## Build

An embeddings instance can be created as follows:

```python
embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
```

The example above builds a transformers-based embeddings instance. In this case, when loading and searching for data, a transformers model is used to vectorize data.

The embeddings instance is [configuration-driven](configuration) based on what is passed in the constructor. Embeddings indexes store vectors and can optionally [store content](configuration#content). Content storage enables additional filtering and data retrieval options.

## Index

After creating a new embeddings instance, the next step is adding data to it.

```python
embeddings.index([(uid, text, None) for uid, text in enumerate(data)])
```

The index method takes an iterable collection of tuples with three values. 

| Element     | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| id          | unique record id                                              |
| data        | input data to index, can be text, a dictionary or object      |
| tags        | optional tags string, used to mark/label data as it's indexed |

When the data element is a dictionary and it has a field named `text`, that will be used for indexing.

The input iterable can be a list or generator. [Generators](https://wiki.python.org/moin/Generators) help with indexing very large datasets as only portions of the data is in memory at any given time.

More information on indexing can be found in the [index guide](indexing).

## Search

Once data is indexed, it is ready for search.

```python
embeddings.search(query, limit)
```

The search method takes two parameters, the query and query limit. The results format is different based on whether [content](configuration/#content) is stored or not.

- List of `(id, score)` when content is _not_ stored
- List of `{**query columns}` when content is stored

Both natural language and SQL queries are supported. More information can be found in the [query guide](query).

## More examples

See [this link](../examples/#semantic-search) for a full list of embeddings examples.
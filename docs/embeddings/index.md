# Embeddings

![embeddings](../images/embeddings.png#only-light)
![embeddings](../images/embeddings-dark.png#only-dark)

Embeddings databases are the engine that delivers semantic search. Data is transformed into embeddings vectors where similar concepts will produce similar vectors. Indexes both large and small are built with these vectors. The indexes are used to find results that have the same meaning, not necessarily the same keywords.

The following code snippet shows how to build and search an embeddings index.

```python
from txtai import Embeddings

# Create embeddings model, backed by sentence-transformers & transformers
embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2")

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

# Index the list of text
embeddings.index(data)

print(f"{'Query':20} Best Match")
print("-" * 50)

# Run an embeddings search for each query
for query in ("feel good story", "climate change", "public health story", "war",
              "wildlife", "asia", "lucky", "dishonest junk"):
    # Extract uid of first result
    # search result format: (uid, score)
    uid = embeddings.search(query, 1)[0][0]

    # Print text
    print(f"{query:20} {data[uid]}")
```

## Build

An embeddings instance is [configuration-driven](configuration) based on what is passed in the constructor. Vectors are stored with the option to also [store content](configuration/database#content). Content storage enables additional filtering and data retrieval options.

The example above sets a specific embeddings vector model via the [path](configuration/vectors/#path) parameter. An embeddings instance with no configuration can also be created.

```python
embeddings = Embeddings()
```

In this case, when loading and searching for data, the [default transformers vector model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is used to vectorize data. See the [model guide](../models) for current model recommentations.

## Index

After creating a new embeddings instance, the next step is adding data to it.

```python
embeddings.index(rows)
```

The index method takes an iterable and supports the following formats for each element.

- `(id, data, tags)` - default processing format

| Element     | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| id          | unique record id                                              |
| data        | input data to index, can be text, a dictionary or object      |
| tags        | optional tags string, used to mark/label data as it's indexed |

- `(id, data)`

  Same as above but without tags.

- `data`

Single element to index. In this case, unique id's will automatically be generated. Note that for generated id's, [upsert](methods/#txtai.embeddings.base.Embeddings.upsert) and [delete](methods/#txtai.embeddings.base.Embeddings.delete) calls require a separate search to get the target ids.

When the data field is a dictionary, text is passed via the `text` key, binary objects via the `object` key. Note that [content](configuration/database#content) must be enabled to store metadata and [objects](configuration/database#objects) to store binary object data. The `id` and `tags` keys will be extracted, if provided.

The input iterable can be a list or generator. [Generators](https://wiki.python.org/moin/Generators) help with indexing very large datasets as only portions of the data is in memory at any given time.

More information on indexing can be found in the [index guide](indexing).

## Search

Once data is indexed, it is ready for search.

```python
embeddings.search(query, limit)
```

The search method takes two parameters, the query and query limit. The results format is different based on whether [content](configuration/database#content) is stored or not.

- List of `(id, score)` when content is _not_ stored
- List of `{**query columns}` when content is stored

Both natural language and SQL queries are supported. More information can be found in the [query guide](query).

## Resource management

Embeddings databases are context managers. The following blocks automatically [close](methods/#txtai.embeddings.base.Embeddings.close) and free resources upon completion.

```python
# Create a new Embeddings database, index data and save
with Embeddings() as embeddings:
  embeddings.index(rows)
  embeddings.save(path)

# Search a saved Embeddings database
with Embeddings().load(path) as embeddings:
  embeddings.search(query)
```

While calling `close` isn't always necessary (resources will be garbage collected), it's best to free shared resources like database connections as soon as they aren't needed.

## More examples

See [this link](../examples/#semantic-search) for a full list of embeddings examples.
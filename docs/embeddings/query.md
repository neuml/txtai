# Query guide

![query](../images/query.png#only-light)
![query](../images/query-dark.png#only-dark)

This section covers how to query data with txtai. The simplest way to search for data is building a natural language string with the desired content to find. txtai also supports querying with SQL. We'll cover both methods here.

## Natural language queries

In the simplest case, the query is text and the results are index text that is most similar to the query text.

```python
embeddings.search("feel good story")
embeddings.search("wildlife")
```

The queries above [search](../methods#txtai.embeddings.base.Embeddings.search) the index for similarity matches on `feel good story` and `wildlife`. If content storage is enabled, a list of `{**query columns}` is returned. Otherwise, a list of `(id, score)` tuples are returned.

## SQL

txtai supports more complex queries with SQL. This is only supported if [content storage](../configuration/database#content) is enabled. txtai has a translation layer that analyzes input SQL statements and combines similarity results with content stored in a relational database.

SQL queries are run through `embeddings.search` like natural language queries but the examples below only show the SQL query for conciseness.

```python
embeddings.search("SQL query")
```

### Similar clause

The similar clause is a txtai function that enables similarity searches with SQL.

```sql
SELECT id, text, score FROM txtai WHERE similar('feel good story')
```

The similar clause takes the following arguments:

```sql
similar("query", "number of candidates", "index", "weights")
```

| Argument              | Description                            |
| --------------------- | ---------------------------------------|
| query                 | natural language query to run          |
| number of candidates  | number of candidate results to return  |
| index                 | target index name                      |
| weights               | hybrid score weights                   |

The txtai query layer joins results from two separate components, a relational store and a similarity index. With a similar clause, a similarity search is run and those ids are fed to the underlying database query.

The number of candidates should be larger than the desired number of results when applying additional filter clauses. This ensures that `limit` results are still returned after applying additional filters. If the number of candidates is not specified, it is defaulted as follows:

- For a single query filter clause, the default is the query limit
- With multiple filtering clauses, the default is 10x the query limit

The index name is only applicable when [subindexes](../configuration/general/#indexes) are enabled. This specifies the index to use for the query.

Weights sets the hybrid score weights when an index has both a sparse and dense index.

### Dynamic columns

Content can be indexed in multiple ways when content storage is enabled. [Remember that input documents](../#index) take the form of `(id, data, tags)` tuples. If data is a string or binary content, it's indexed and searchable with `similar()` clauses.

If data is a dictionary, then all fields in the dictionary are stored and available via SQL. The `text` field or [field specified in the index configuration](../configuration/general/#columns) is indexed and searchable with `similar()` clauses.

For example:

```python
embeddings.index([{"text": "text to index", "flag": True,
                   "actiondate": "2022-01-01"}])
```

With the above input data, queries can now have more complex filters.

```sql
SELECT text, flag, actiondate FROM txtai WHERE similar('query') AND flag = 1
AND actiondate >= '2022-01-01'
```

txtai's query layer automatically detects columns and translates queries into a format that can be understood by the underlying database.

Nested dictionaries/JSON is supported and can be escaped with bracket statements.

```python
embeddings.index([{"text": "text to index",
                   "parent": {"child element": "abc"}}])
```

```sql
SELECT text FROM txtai WHERE [parent.child element] = 'abc'
```

Note the bracket statement escaping the nested column with spaces in the name.

### Bind parameters

txtai has support for SQL bind parameters.

```python
# Query with a bind parameter for similar clause
query = "SELECT id, text, score FROM txtai WHERE similar(:x)"
results = embeddings.search(query, parameters={"x": "feel good story"})

# Query with a bind parameter for column filter
query = "SELECT text, flag, actiondate FROM txtai WHERE flag = :x"
results = embeddings.search(query, parameters={"x": 1})
```

### Aggregation queries

The goal of txtai's query language is to closely support all functions in the underlying database engine. The main challenge is ensuring dynamic columns are properly escaped into the engines native query function. 

Aggregation query examples.

```sql
SELECT count(*) FROM txtai WHERE similar('feel good story') AND score >= 0.15
SELECT max(length(text)) FROM txtai WHERE similar('feel good story')
AND score >= 0.15
SELECT count(*), flag FROM txtai GROUP BY flag ORDER BY count(*) DESC
```

## Binary objects

txtai has support for storing and retrieving binary objects. Binary objects can be retrieved as shown in the example below.

```python
# Create embeddings index with content and object storage enabled
embeddings = Embeddings(content=True, objects=True)

# Get an image
request = open("demo.gif", "rb")

# Insert record
embeddings.index([("txtai", {"text": "txtai executes machine-learning workflows.",
                             "object": request.read()})])

# Query txtai and get associated object
query = "SELECT object FROM txtai WHERE similar('machine learning') LIMIT 1"
result = embeddings.search(query)[0]["object"]

# Query binary content with a bind parameter
query = "SELECT object FROM txtai WHERE similar(:x) LIMIT 1"
results = embeddings.search(query, parameters={"x": request.read()})
```

## Custom SQL functions

Custom, user-defined SQL functions extend selection, filtering and ordering clauses with additional logic. For example, the following snippet defines a function that translates text using a translation pipeline.

```python
# Translation pipeline
translate = Translation()

# Create embeddings index
embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2",
                        content=True,
                        functions=[translate]})

# Run a search using a custom SQL function
embeddings.search("""
SELECT
  text,
  translation(text, 'de', null) 'text (DE)',
  translation(text, 'es', null) 'text (ES)',
  translation(text, 'fr', null) 'text (FR)'
FROM txtai WHERE similar('feel good story')
LIMIT 1
""")
```

## Query translation

Natural language queries with filters can be converted to txtai-compatible SQL statements with query translation. For example:

```python
embeddings.search("feel good story since yesterday")
```

can be converted to a SQL statement with a similar clause and date filter.

```sql
select id, text, score from txtai where similar('feel good story') and
entry >= date('now', '-1 day')
```

This requires setting a [query translation model](../configuration/database#query). The default query translation model is [t5-small-txtsql](https://huggingface.co/NeuML/t5-small-txtsql) but this can easily be finetuned to handle different use cases.

## Hybrid search

When an embeddings database has both a sparse and dense index, both indexes will be queried and the results will be equally weighted unless otherwise specified.

```python
embeddings.search("query", weights=0.5)
embeddings.search("SELECT id, text, score FROM txtai WHERE similar('query', 0.5)")
```

## Graph search

If an embeddings database has an associated graph network, graph searches can be run. The search syntax below uses [openCypher](https://github.com/opencypher/openCypher). Follow the preceding link to learn more about this syntax.

Additionally, standard embeddings searches can be returned as graphs.

```python
# Find all paths between id: 0 and id: 5 between 1 and 3 hops away
embeddings.graph.search("""
MATCH P=({id: 0})-[*1..3]->({id: 5})
RETURN P
""")

# Standard embeddings search as graph
embeddings.search("query", graph=True)
```

## Subindexes

Subindexes can be queried as follows:

```python
# Query with index parameter
embeddings.search("query", index="subindex1")

# Specify with SQL
embeddings.search("""
SELECT id, text, score FROM txtai
WHERE similar('query', 'subindex1')
""")
```

## Combined index architecture

txtai has multiple storage and indexing components. Content is stored in an underlying database along with an approximate nearest neighbor (ANN) index, keyword index and graph network. These components combine to deliver similarity search alongside traditional structured search.

The ANN index stores ids and vectors for each input element. When a natural language query is run, the query is translated into a vector and a similarity query finds the best matching ids. When a database is added into the mix, an additional step is executed. This step takes those ids and effectively inserts them as part of the underlying database query. The same steps apply with keyword indexes except a term frequency index is used to find the best matching ids.

Dynamic columns are supported via the underlying engine. For SQLite, data is stored as JSON and dynamic columns are converted into `json_extract` clauses. Client-server databases are supported via [SQLAlchemy](https://docs.sqlalchemy.org/en/20/dialects/) and dynamic columns are supported provided the underlying engine has [JSON](https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.JSON) support.

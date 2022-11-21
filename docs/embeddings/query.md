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

txtai supports more complex queries with SQL. This is only supported if [content storage](../configuration/#content) is enabled. txtai has a translation layer that analyzes input SQL statements and combines similarity results with content stored in a relational database.

SQL queries are run through `embeddings.search` like natural language queries but the examples below only show the SQL query for conciseness.

```python
embeddings.search("SQL query")
```

### Similar clause

The similar clause is a txtai function that enables similarity searches with SQL.

```sql
SELECT id, text, score FROM txtai WHERE similar('feel good story')
SELECT id, text, score FROM txtai WHERE similar('feel good story')
```

The similar clause takes two arguments:

```sql
similar("query", "number of candidates")
```

| Argument              | Description                            |
| --------------------- | ---------------------------------------|
| query                 | natural language query to run          |
| number of candidates  | number of candidate results to return  |

The txtai query layer has to join results from two separate components, a relational store and a similarity index. With a similar clause, a similarity search is run and those ids are fed to the underlying database query.

The number of candidates should be larger than the desired number of results when applying additional filter clauses. This ensures that `limit` results are still returned after applying additional filters. If the number of candidates is not specified, it is defaulted as follows:

- For a single query filter clause, the default is the query limit
- With multiple filtering clauses, the default is 10x the query limit

### Dynamic columns

Content can be indexed in multiple ways when content storage is enabled. [Remember that input documents](../#index) take the form of `(id, data, tags)` tuples. If data is a string, then content is primarily filtered with similar clauses. If data is a dictionary, then all fields in the dictionary are indexed and searchable.

For example:

```python
embeddings.index([(0, {"text": "text to index", "flag": True,
                       "entry": "2022-01-01"}, None)])
```

With the above input data, queries can now have more complex filters.

```sql
SELECT text, flag, entry FROM txtai WHERE similar('query') AND flag = 1
AND entry >= '2022-01-01'
```

txtai's query layer automatically detects columns and translates queries into a format that can be understood by the underlying database.

Nested dictionaries/JSON is supported and can be escaped with bracket statements.

```python
embeddings.index([(0, {"text": "text to index",
                       "parent": {"child element": "abc"}}, None)])
```

```sql
SELECT text FROM txtai WHERE [parent.child element] ='abc'
```

Note the bracket statement escaping the nested column with spaces in the name.

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
# Get an image
request = open("demo.gif")

# Insert record
embeddings.index([("txtai", {"text": "txtai executes machine-learning workflows.",
                             "object": request.read()}, None)])

# Query txtai and get associated object
query = "select object from txtai where similar('machine learning') limit 1"
result = embeddings.search(query)[0]["object"]
```

## Custom SQL functions

Custom, user-defined SQL functions extend selection, filtering and ordering clauses with additional logic. For example, the following snippet defines a function that translates text using a translation pipeline.

```python
# Translation pipeline
translate = Translation()

# Create embeddings index
embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2",
                         "content": True,
                         "functions": [translate]})

# Run a search using a custom SQL function
embeddings.search("""
select
  text,
  translation(text, 'de', null) 'text (DE)',
  translation(text, 'es', null) 'text (ES)',
  translation(text, 'fr', null) 'text (FR)'
from txtai where similar('feel good story')
limit 1
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

This requires setting a [query translation model](../configuration#query). The default query translation model is [t5-small-txtsql](https://huggingface.co/NeuML/t5-small-txtsql) but this can easily be finetuned to handle different use cases.

## Combined index architecture

When content storage is enabled, txtai becomes a dual storage engine. Content is stored in an underlying database (currently supports SQLite) along with an Approximate Nearest Neighbor (ANN) index. These components combine to deliver similarity search alongside traditional structured search.

The ANN index stores ids and vectors for each input element. When a natural language query is run, the query is translated into a vector and a similarity query finds the best matching ids. When a database is added into the mix, an additional step is applied. This step takes those ids and effectively inserts them as part of the underlying database query.

Dynamic columns are supported via the underlying engine. For SQLite, data is stored as JSON and dynamic columns are converted into `json_extract` clauses. This same concept can be expanded to other storage engines like PostgreSQL and could even work with NoSQL stores. 

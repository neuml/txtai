# Database

The following covers available content storage configuration options.

## content
```yaml
content: boolean|sqlite|duckdb|custom
```

Enables content storage. When true, the default storage engine, `sqlite` will be used to save metadata alongside embeddings vectors. Also supports `duckdb`. Add custom storage engines via setting this parameter to the fully resolvable class string.

Content storage specific settings are set with a corresponding configuration object having the same name as the content storage engine (i.e. duckdb or sqlite). None of these are required and are set to defaults if omitted.

### sqlite
```yaml
sqlite:
    wal: enable write-ahead logging - allows concurrent read/write operations,
         defaults to false
```

## objects
```yaml
objects: boolean
```

Enables object storage. When content storage is enabled and this is true, support for storing binary content alongside embeddings vectors and metadata is enabled.

## functions
```yaml
functions: list
```

List of functions with user-defined SQL functions, only used when [content](#content) is enabled. Each list element must be one of the following:

- function
- callable object
- dict with fields for name, argcount and function

[An example can be found here](../../query#custom-sql-functions).

## query
```yaml
query:
    path: sets the path for the query model - this can be any model on the
          Hugging Face Model Hub or a local file path.
    prefix: text prefix to prepend to all inputs
    maxlength: maximum generated sequence length
```

Query translation model. Translates natural language queries to txtai compatible SQL statements.

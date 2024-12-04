# Database

Databases store metadata, text and binary content.

## content
```yaml
content: boolean|sqlite|duckdb|client|url|custom
```

Enables content storage. When true, the default storage engine, `sqlite` will be used to save metadata.

Client-server connections are supported with either `client` or a full connection URL. When set to `client`, the CLIENT_URL environment variable must be set to the full connection URL. See the [SQLAlchemy](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) documentation for more information on how to construct connection strings for client-server databases.

Add custom storage engines via setting this parameter to the fully resolvable class string.

Content storage specific settings are set with a corresponding configuration object having the same name as the content storage engine (i.e. duckdb or sqlite). These are optional and set to defaults if omitted.

### client
```yaml
schema:  default database schema for the session - defaults to being
         determined by the database
```

Additional settings for client-server databases. Also supported when the `content=url`.

### sqlite
```yaml
sqlite:
    wal: enable write-ahead logging - allows concurrent read/write operations,
         defaults to false
```

Additional settings for SQLite.

## objects
```yaml
objects: boolean|image|pickle
```

Enables object storage. Supports storing binary content. Requires content storage to also be enabled.

Object encoding options are:

- `standard`: Default encoder when boolean set. Encodes and decodes objects as byte arrays.
- `image`: Image encoder. Encodes and decodes objects as image objects.
- `pickle`: Pickle encoder. Encodes and decodes objects with the pickle module. Supports arbitrary objects.

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

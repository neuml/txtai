# Cloud

This section describes parameters used to sync compressed indexes with cloud storage. These parameters are only enabled if an embeddings index is stored as compressed. They are set via the [embeddings.load](../../methods/#txtai.embeddings.base.Embeddings.load) and [embeddings.save](../../methods/#txtai.embeddings.base.Embeddings.save) methods.

## provider
```yaml
provider: string
```

The cloud storage provider, see [full list of providers here](https://libcloud.readthedocs.io/en/stable/storage/supported_providers.html).

## container
```yaml
container: string
```

Container/bucket/directory name.

## key
```yaml
key: string
```

Provider-specific access key. Can also be set via ACCESS_KEY environment variable. Ensure the configuration file is secured if added to the file.

## secret
```yaml
secret: string
```

Provider-specific access secret. Can also be set via ACCESS_SECRET environment variable. Ensure the configuration file is secured if added to the file.

## host
```yaml
host: string
```

Optional server host name. Set when using a local cloud storage server.

## port
```yaml
port: int
```

Optional server port. Set when using a local cloud storage server.

## token
```yaml
token: string
```

Optional temporary session token

## region
```yaml
region: string
```

Optional parameter to specify the storage region, provider-specific.

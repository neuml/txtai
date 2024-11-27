# Cloud

The following describes parameters used to sync indexes with cloud storage. Cloud object storage, the [Hugging Face Hub](https://huggingface.co/models) and custom providers are all supported.

Parameters are set via the [embeddings.load](../../methods/#txtai.embeddings.base.Embeddings.load) and [embeddings.save](../../methods/#txtai.embeddings.base.Embeddings.save) methods.

## provider
```yaml
provider: string
```

Cloud provider. Can be one of the following:

- Cloud object storage. Set to one of these [providers](https://libcloud.readthedocs.io/en/stable/storage/supported_providers.html). Use the text shown in the `Provider Constant` column as lower case.

- Hugging Face Hub. Set to `huggingface-hub`.

- Custom providers. Set to the full class path of the custom provider.

## container
```yaml
container: string
```

Container/bucket/directory/repository name. Embeddings will be stored in the container with the filename specified by the `path` configuration.

## Cloud object storage configuration

In addition to the above common configuration, the cloud object storage provider has the following additional configuration parameters. Note that some cloud providers do not need any of these parameters and can use implicit authentication with service accounts.

See the [libcloud documentation](https://libcloud.readthedocs.io/en/stable/apidocs/libcloud.common.html#module-libcloud.common.base) for more information on these parameters.

### key
```yaml
key: string
```

Provider-specific access key. Can also be set via `ACCESS_KEY` environment variable. Ensure the configuration file is secured if added to the file. When using implicit authentication, set this to a value such as 'using-implicit-auth'.

### secret
```yaml
secret: string
```

Provider-specific access secret. Can also be set via `ACCESS_SECRET` environment variable. Ensure the configuration file is secured if added to the file. When using implicit authentication, this option is not required.

### prefix
```yaml
prefix: string
```

Optional object prefix. Object storage doesn't have the concept of a directory but a prefix is similar. For example, a prefix could be `base/dir`. This helps with organizing data in an object storage bucket.

More can be found at the following links.

- [Organizing objects using prefixes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-prefixes.html)
- [libcloud container method documentation](https://libcloud.readthedocs.io/en/stable/storage/api.html#libcloud.storage.base.StorageDriver.iterate_container_objects)

### host
```yaml
host: string
```

Optional server host name. Set when using a local cloud storage server.

### port
```yaml
port: int
```

Optional server port. Set when using a local cloud storage server.

### token
```yaml
token: string
```

Optional temporary session token

### region
```yaml
region: string
```

Optional parameter to specify the storage region, provider-specific.

## Hugging Face Hub configuration

The huggingface-hub provider supports the following additional configuration parameters. More on these parameters can be found in the [Hugging Face Hub's documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/overview).

### revision
```yaml
revision: string
```

Optional Git revision id which can be a branch name, a tag, or a commit hash

### cache
```yaml
cache: string
```

Path to the folder where cached files are stored

### token
```yaml
token: string|boolean
```

Token to be used for the download. If set to True, the token will be read from the Hugging Face config folder.

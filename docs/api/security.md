# Security

The default implementation of an API service runs via HTTP and is fully open. If the service is being run as a prototype on an internal network, that may be fine. In most scenarios, the connection should at least be encrypted. Authorization is another built-in feature that requires a valid API token with each request. See below for more.

## HTTPS

The default API service command starts a Uvicorn server as a HTTP service on port 8000. To run a HTTPS service, consider the following options.

- [TLS Proxy Server](https://fastapi.tiangolo.com/deployment/https/). *Recommended choice*. With this configuration, the txtai API service runs as a HTTP service only accessible on the localhost/local network. The proxy server handles all encryption and redirects requests to local services. See this [example configuration](https://www.uvicorn.dev/deployment/#running-behind-nginx) for more.

- [Uvicorn SSL Certificate](https://www.uvicorn.dev/deployment/). Another option is setting the SSL certificate on the Uvicorn service. This works in simple situations but gets complex when hosting multiple txtai or other related services.

## Authorization

Authorization requires a valid API token with each API request. This token is sent as a HTTP `Authorization` header. 

*Server*
```bash
CONFIG=config.yml TOKEN=<sha256 encoded token> uvicorn "txtai.api:app"
```

*Client*
```bash
curl \
  -X POST "http://localhost:8000/workflow" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \ 
  -d '{"name":"sumfrench", "elements": ["https://github.com/neuml/txtai"]}'
```

It's important to note that HTTPS **must** be enabled using one of the methods mentioned above. Otherwise, tokens will be exchanged as clear text. 

Authentication and Authorization can be fully customized. See the [dependencies](../customization#dependencies) section for more.

## Safe Open

Many components in txtai work with local files and have no restrictions on what files and urls can be accessed. The [textractor pipeline](../../pipeline/data/textractor) and [retrieve](../../workflow/task/retrieve) tasks both have a built-in option called `safeopen` which is enabled by default when running the API and applications. It enforces the following rules:

- Local files must be in the safeopen directory (defaults to temp dir)
- URLs must be public URLs (also has checks for redirects and DNS rebind attempts)

The following workflow only allows accessing files stored in a specific directory along with public urls.

```yaml
workflow:
  textract:
    tasks:
      - action: textractor
```

A number of of pipelines support reading local files. For these it's recommended to wrap those pipelines as workflows with a `retrieve` task. Note that pipelines can be created in-line and don't need to be exposed via the API.

```yaml
workflow:
  caption:
    tasks:
      - task: retrieve
        action: caption
```

Additionally, internal pipelines can be created but not exposed via the API.

```yaml
routes:
  caption: False

caption:

workflow:
  caption:
    tasks:
      - task: retrieve
        action: caption
```

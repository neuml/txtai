# Model Context Protocol

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open standard that enables developers to build secure, two-way connections between their data sources and AI-powered tools.

The API can be configured to handle MCP requests. All enabled endpoints set in the API configuration are automatically added as MCP tools.

```yaml
mcp: True
```

Once this configuration option is added, a new route is added to the application `/mcp`. 

The [Model Context Protocol Inspector tool](https://www.npmjs.com/package/@modelcontextprotocol/inspector) is a quick way to explore how the MCP tools are exported through this interface.

Run the following and go to the local URL specified.

```
npx @modelcontextprotocol/inspector node build/index.js
```

Enter `http://localhost:8000/mcp` to see the full list of tools available.

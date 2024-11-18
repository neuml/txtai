# Configuration

Configuration is set through YAML. In most cases, YAML keys map to fields names in Python. The [example in the previous section](../) gave a full-featured example covering a wide array of configuration options.

Each section below describes the available configuration settings.

## Embeddings

The configuration parser expects a top level `embeddings` key to be present in the YAML. All [embeddings configuration](../../embeddings/configuration) is supported.

The following example defines an embeddings index.

```yaml
path: index path
writable: true

embeddings:
  path: vector model
  content: true
```

Three top level settings are available to control where indexes are saved and if an index is a read-only index.

### path
```yaml
path: string
```

Path to save and load the embeddings index. Each API instance can only access a single index at a time.

### writable
```yaml
writable: boolean
```

Determines if the input embeddings index is writable (true) or read-only (false). This allows serving a read-only index.

### cloud
[Cloud storage settings](../../embeddings/configuration/cloud) can be set under a `cloud` top level configuration group.

## Agent

Agents are defined under a top level `agent` key. Each key under the `agent` key is the name of the agent. Constructor parameters can be passed under this key.

The following example defines an agent.

```yaml
agent:
    researcher:
        tools:
            - websearch

llm:
    path: hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

## Pipeline

Pipelines are loaded as top level configuration parameters. Pipeline names are automatically detected in the YAML configuration and created upon startup. All [pipelines](../../pipeline) are supported.

The following example defines a series of pipelines. Note that entries below are the lower-case names of the pipeline class.

```yaml
caption:

extractor:
  path: model path

labels:

summary:

tabular:

translation:
```

Under each pipeline name, configuration settings for the pipeline can be set.

## Workflow

Workflows are defined under a top level `workflow` key. Each key under the `workflow` key is the name of the workflow. Under that is a `tasks` key with each task definition.

The following example defines a workflow.

```yaml
workflow:
  sumtranslate:
    tasks:
        - action: summary
        - action: translation
```

### schedule

Schedules a workflow using a [cron expression](../../workflow/schedule).

```yaml
workflow:
  index:
    schedule:
      cron: 0/10 * * * * *
      elements: ["api params"] 
    tasks:
      - task: service
        url: api url
      - action: index
```

### tasks
```yaml
tasks: list
```

Expects a list of workflow tasks. Each element defines a single workflow task. All [task configuration](../../workflow/task) is supported.

A shorthand syntax for creating tasks is supported. This syntax will automatically map task strings to an `action:value` pair.

Example below.

```yaml
workflow:
  index:
    tasks:
      - action1
      - action2
```

Each task element supports the following additional arguments.

#### action
```yaml
action: string|list
```

Both single and multi-action tasks are supported.

The action parameter works slightly different when passed via configuration. The parameter(s) needs to be converted into callable method(s). If action is a pipeline that has been defined in the current configuration, it will use that pipeline as the action.

There are three special action names `index`, `upsert` and `search`. If `index` or `upsert` are used as the action, the task will collect workflow data elements and load them into defined the embeddings index. If `search` is used, the task will execute embeddings queries for each input data element.

Otherwise, the action must be a path to a callable object or function. The configuration parser will resolve the function name and use that as the task action.

#### task
```yaml
task: string
```

Optionally sets the type of task to create. For example, this could be a `file` task or a `retrieve` task. If this is not specified, a generic task is created. [The list of workflow tasks can be found here](../../workflow).

#### args
```yaml
args: list
```

Optional list of static arguments to pass to the workflow task. These are combined with workflow data to pass to each `__call__`.

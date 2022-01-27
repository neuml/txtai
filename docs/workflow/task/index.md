# Tasks

![task](../../images/task.png#only-light)
![task](../../images/task-dark.png#only-dark)

Workflows execute tasks. Tasks are callable objects with a number of parameters to control the processing of data at a given step. While similar to pipelines, tasks encapsulate processing and don't perform signficant transformations on their own. Tasks perform logic to prepare content for the underlying action(s).

A simple task is shown below.

```python
Task(lambda x: [y * 2 for y in x])
```

The task above executes the function above for all input elements.

Tasks work well with pipelines, since pipelines are callable objects. The example below will summarize each input element.

```python
summary = Summary()
Task(summary)
```

Tasks can operate independently but work best with workflows, as workflows add large-scale stream processing.

```python
summary = Summary()
task = Task(summary)
task(["Very long text here"])

workflow = Workflow([task])
list(workflow(["Very long text here"]))
```

Tasks can also be created with configuration as part of a workflow.

```yaml
workflow:
  tasks:
    - action: summary 
```

::: txtai.workflow.Task.__init__

## Multi-action task concurrency

The default processing mode is to run actions sequentially. Multiprocessing support is already built in at a number of levels. Any of the GPU models will maximize GPU utilization for example and even in CPU mode, concurrency is utilized. But there are still use cases for task action concurrency. For example, if the system has multiple GPUs, the task runs external sequential code, or the task has a large number of I/O tasks.

In addition to sequential processing, multi-action tasks can run either multithreaded or with multiple processes. The advantages of each approach are discussed below.

- *multithreading* - no overhead of creating separate processes or pickling data. But Python can only execute a single thread due the GIL, so this approach won't help with CPU bound actions. This method works well with I/O bound actions and GPU actions.

- *multiprocessing* - separate subprocesses are created and data is exchanged via pickling. This method can fully utilize all CPU cores since each process runs independently. This method works well with CPU bound actions.

More information on multiprocessing can be found in the [Python documentation](https://docs.python.org/3/library/multiprocessing.html).

## Multi-action task merges

Multi-action tasks will generate parallel outputs for the input data. The task output can be merged together in a couple different ways.

### ::: txtai.workflow.Task.hstack
### ::: txtai.workflow.Task.vstack
### ::: txtai.workflow.Task.concat

## Extract task output columns

With column-wise merging, each output row will be a tuple of output values for each task action. This can be fed as input to a downstream task and that task can have separate tasks work with each element.

A simple example:

```python
workflow = Workflow([Task(lambda x: [y * 3 for y in x], unpack=False, column=0)])
list(workflow([(2, 8)]))
```

For the example input tuple of (2, 2), the workflow will only select the first element (2) and run the task against that element. 

```python
workflow = Workflow([Task([lambda x: [y * 3 for y in x], 
                           lambda x: [y - 1 for y in x]],
                           unpack=False, column={0:0, 1:1})])
list(workflow([(2, 8)]))
```

The example above applies a separate action to each input column. This simple construct can help build extremely powerful workflow graphs!

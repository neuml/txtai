# Tasks

Workflows execute tasks. Tasks are callable objects with a number of parameters to control the processing of data at a given step. 

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

::: txtai.workflow.Task.__init__

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

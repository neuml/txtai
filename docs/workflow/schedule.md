# Schedule

![schedule](../images/schedule.png#only-light)
![schedule](../images/schedule-dark.png#only-dark)

Workflows can run on a repeating basis with schedules. This is suitable in cases where a workflow is run against a dynamically expanding input, like an API service or directory of files. 

The schedule method takes a cron expression, list of static elements (which dynamically expand i.e. API service, directory listing) and an optional maximum number of iterations.

Below are a couple example cron expressions.

```bash
# ┌─────────────── minute (0 - 59)
# | ┌───────────── hour (0 - 23)
# | | ┌─────────── day of the month (1 - 31)
# | | | ┌───────── month (1 - 12)
# | | | | ┌─────── day of the week (0 - 6)
# | | | | | ┌───── second (0 - 59)
# | | | | | |
  * * * * * *      # Run every second
0/5 * * * *        # Run every 5 minutes
  0 0 1 * *        # Run monthly on 1st
  0 0 1 1 *        # Run on Jan 1 at 12am
  0 0 * * mon,wed  # Run Monday and Wednesday
```

## Python
Simple workflow scheduled with Python. See [schedule](../#txtai.workflow.base.Workflow.schedule) method definition below.

```python
workflow = Workflow(tasks)
workflow.schedule("0/5 * * * *", elements)
```

## Configuration 
Simple workflow scheduled with configuration.

```yaml
workflow:
  index:
    schedule:
      cron: 0/5 * * * *
      elements: [...]
    tasks: [...]
```

```python
# Create and run the workflow
from txtai.api import API

# Create and run the workflow
app = API("workflow.yml")

# Wait for scheduled workflows
app.wait()
```

See the links below for more information on cron expressions.

- [cron overview](https://en.wikipedia.org/wiki/Cron)
- [croniter - library used by txtai](https://github.com/kiorky/croniter)

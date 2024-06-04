# FAQ

![faq](images/faq.png)

Below is a list of frequently asked questions and common issues encountered.

## Questions

----------

__Question__

What models are recommended?

__Answer__

See the [model guide](../models).

----------

__Question__

What is the best way to track the progress of an `embeddings.index` call?

__Answer__

Wrap the list or generator passed to the index call with tqdm. See [#478](https://github.com/neuml/txtai/issues/478) for more.

----------

__Question__

What is the best way to analyze the content of a txtai index?

__Answer__

txtai has a console application that makes this easy. Read [this article](https://medium.com/neuml/insights-from-the-txtai-console-d307c28e149e) to learn more.

----------

__Question__

How can models be externally loaded and passed to embeddings and pipelines?

__Answer__

Embeddings example.

```python
from transformers import AutoModel, AutoTokenizer
from txtai.embeddings import Embeddings

# Load model externally
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Pass to embeddings instance
embeddings = Embeddings(path=model, tokenizer=tokenizer)
```

LLM pipeline example.

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from txtai.pipeline import LLM

# Load Mistral-7B-OpenOrca
path = "Open-Orca/Mistral-7B-OpenOrca"
model = AutoModelForCausalLM.from_pretrained(
  path,
  torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(path)

llm = LLM((model, tokenizer))
```

## Common issues

----------

__Issue__

Embeddings query errors like this:

```
SQLError: no such function: json_extract
```

__Solution__

Upgrade Python version as it doesn't have SQLite support for `json_extract`

----------

__Issue__

Segmentation faults and similar errors on macOS

__Solution__

Set the following environment parameters.

- Disable OpenMP threading via the environment variable `export OMP_NUM_THREADS=1`
- Disable PyTorch MPS device via `export PYTORCH_MPS_DISABLE=1`
- Disable llama.cpp metal via `export LLAMA_NO_METAL=1`

If the issue is with a training pipeline.

- Disable GPU training via `export ACCELERATE_USE_CPU=1`

----------

__Issue__

`ContextualVersionConflict` and/or package METADATA exception while running one of the [examples](../examples) notebooks on Google Colab

__Solution__

Restart the kernel. See issue [#409](https://github.com/neuml/txtai/issues/409) for more on this issue. 

----------

__Issue__

Error installing optional/extra dependencies such as `pipeline`

__Solution__

The default MacOS shell (zsh) and Windows PowerShell require escaping square brackets

```
pip install 'txtai[pipeline]'
```

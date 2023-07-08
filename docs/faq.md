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
embeddings = Embeddings({"path": model, "tokenizer": tokenizer})
```

LLM pipeline example.

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from txtai.pipeline import LLM

# Load Falcon-7B-Instruct
path = "tiiuae/falcon-7b-instruct"
model = AutoModelForCausalLM.from_pretrained(
  path,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True
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

Upgrade Python version as it doesn't have SQLite support for json_extract

----------

__Issue__

Segmentation faults and similar errors on macOS

__Solution__

Downgrade PyTorch to <= 1.12. See issue [#377](https://github.com/neuml/txtai/issues/377) for more on this issue. 

----------

__Issue__

`ContextualVersionConflict` exception when importing certain libraries while running one of the [examples](../examples) notebooks on Google Colab

__Solution__

Restart the kernel. See issue [#409](https://github.com/neuml/txtai/issues/409) for more on this issue. 

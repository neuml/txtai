# HFTrainer

Trains a new Hugging Face Transformer model using the Trainer framework.

Examples on how to use the trainer below.

```python
import pandas as pd

from datasets import load_dataset

from txtai.pipeline import HFTrainer

trainer = HFTrainer()

# Pandas DataFrame
df = pd.read_csv("training.csv")
model, tokenizer = trainer("bert-base-uncased", df)

# Hugging Face dataset
ds = load_dataset("glue", "sst2")
model, tokenizer = trainer("bert-base-uncased", ds["train"], columns=("sentence", "label"))

# List of dicts
dt = [{"text": "sentence 1", "label": 0}, {"text": "sentence 2", "label": 1}]]
model, tokenizer = trainer("bert-base-uncased", dt)

# Support additional TrainingArguments
model, tokenizer = trainer("bert-base-uncased", dt, 
                            learning_rate=3e-5, num_train_epochs=5)
```

All [TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments) are supported as function arguments to the trainer call.

::: txtai.pipeline.HFTrainer.__init__
::: txtai.pipeline.HFTrainer.__call__

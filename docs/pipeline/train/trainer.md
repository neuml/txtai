# HFTrainer

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

Trains a new Hugging Face Transformer model using the Trainer framework.

## Example

The following shows a simple example using this pipeline.

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

See the links below for more detailed examples.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Train a text labeler](https://github.com/neuml/txtai/blob/master/examples/16_Train_a_text_labeler.ipynb) | Build text sequence classification models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/16_Train_a_text_labeler.ipynb) |
| [Train without labels](https://github.com/neuml/txtai/blob/master/examples/17_Train_without_labels.ipynb) | Use zero-shot classifiers to train new models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/17_Train_without_labels.ipynb) |
| [Train a QA model](https://github.com/neuml/txtai/blob/master/examples/19_Train_a_QA_model.ipynb) | Build and fine-tune question-answering models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/19_Train_a_QA_model.ipynb) |
| [Train a language model from scratch](https://github.com/neuml/txtai/blob/master/examples/41_Train_a_language_model_from_scratch.ipynb) | Build new language models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/41_Train_a_language_model_from_scratch.ipynb) |

## Training tasks

The HFTrainer pipeline builds and/or fine-tunes models for following training tasks.

| Task | Description |
|:-----|:------------|
| language-generation | Causal language model for text generation (e.g. GPT) |
| language-modeling | Masked language model for general tasks (e.g. BERT) |
| question-answering | Extractive question-answering model, typically with the SQuAD dataset |
| sequence-sequence  | Sequence-Sequence model (e.g. T5) |
| text-classification | Classify text with a set of labels |
| token-detection | ELECTRA-style pre-training with replaced token detection |

## Methods 

Python documentation for the pipeline.

### ::: txtai.pipeline.HFTrainer.__call__

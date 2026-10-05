# HFOnnx

![pipeline](../../images/pipeline.png#only-light)
![pipeline](../../images/pipeline-dark.png#only-dark)

Exports a Hugging Face Transformer model to ONNX. Currently, this works best with classification/pooling/qa models. Work is ongoing for sequence to
sequence models (summarization, transcription, translation).

## Example

The following shows a simple example using this pipeline.

```python
from txtai.pipeline import HFOnnx, Labels

# Model path
path = "distilbert-base-uncased-finetuned-sst-2-english"

# Export model to ONNX
onnx = HFOnnx()
model = onnx(path, "text-classification", "model.onnx", True)

# Run inference and validate
labels = Labels((model, path), dynamic=False)
labels("I am happy")
```

See the link below for a more detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Export and run models with ONNX](https://github.com/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb) | Export models with ONNX, run natively in JavaScript, Java and Rust | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb) |

## Methods 

Python documentation for the pipeline.

### ::: txtai.pipeline.HFOnnx.__call__

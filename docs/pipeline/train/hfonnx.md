# HFOnnx

Exports a Hugging Face Transformer model to ONNX. Currently, this works best with classification/pooling/qa models. Work is ongoing for sequence to
sequence models (summarization, transcription, translation).

Example on how to use the pipeline below.

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

::: txtai.pipeline.HFOnnx.__init__
::: txtai.pipeline.HFOnnx.__call__

# HFOnnx

Exports a Hugging Face Transformer model to ONNX.

Example on how to use the pipeline below.

```python
from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer

from txtai.pipeline import HFOnnx

# Normalize logits using sigmoid function
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

# Export model to ONNX
onnx = HFOnnx()
model = onnx("distilbert-base-uncased-finetuned-sst-2-english", "sequence-classification", "model.onnx", True)

# Build ONNX session
options = SessionOptions()
session = InferenceSession(model, options)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokens = tokenizer(["I am happy"], return_tensors="np")

# Run inference and validate
outputs = session.run(None, dict(tokens))
outputs = sigmoid(outputs[0])
```

::: txtai.pipeline.HFOnnx.__init__
::: txtai.pipeline.HFOnnx.__call__

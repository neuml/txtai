# Pipelines

txtai provides a light wrapper around a couple of the Hugging Face pipelines. All pipelines have the following common parameters.

### path
```yaml
path: string
```

Required path to a Hugging Face model

### quantize
```yaml
quantize: boolean
```

Enables dynamic quantization of the Hugging Face model. This is a runtime setting and doesn't save space. It is used to improve the inference time performance of models.

### gpu
```yaml
gpu: boolean
```

Enables GPU inference.

### model
```yaml
model: Hugging Face pipeline or txtai pipeline
```

Shares the underlying model of the passed in pipeline with this pipeline. This allows having variations of a pipeline without having to store multiple copies of the full model in memory.

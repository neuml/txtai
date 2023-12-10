# Pipeline

![pipeline](../images/pipeline.png#only-light)
![pipeline](../images/pipeline-dark.png#only-dark)

txtai provides a generic pipeline processing framework with the only interface requirement being a `__call__` method. Pipelines are flexible and process various types of data. Pipelines can wrap machine learning models as well as other processes.

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../workflow/#configuration-driven-example) or the [API](../api#local-instance).

## List of pipelines

The following is a list of the default pipelines available in txtai. All pipelines use default models when otherwise not specified. See the [model guide](../models) for the current model recommentations.

- Audio
    - [TextToSpeech](audio/texttospeech)
    - [Transcription](audio/transcription)
- Data Processing
    - [Segmentation](data/segmentation)
    - [Tabular](data/tabular)
    - [Text extraction](data/textractor)
- Image
    - [Caption](image/caption)
    - [Image Hash](image/imagehash)
    - [Objects](image/objects)
- Text
    - [Entity](text/entity)
    - [Extractor](text/extractor)
    - [Labeling](text/labels)
    - [LLM](text/llm)
    - [Similarity](text/similarity)
    - [Summary](text/summary)
    - [Translation](text/translation)
- Training
    - [HF ONNX](train/hfonnx)
    - [ML ONNX](train/mlonnx)
    - [Trainer](train/trainer)

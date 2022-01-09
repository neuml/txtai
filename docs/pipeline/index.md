# Pipeline

![pipeline](/images/pipeline.png#only-light)
![pipeline](/images/pipeline-dark.png#only-dark)

txtai provides a generic pipeline processing framework with the only interface requirement being a `__call__` method. Pipelines are flexible and can
process various types of data. Pipelines can wrap machine learning models as well as other processes.

The following pipeline types are currently supported.

- Audio
    - [Transcription](../audio/transcription)
- Data Processing
    - [Segmentation](../data/segmentation)
    - [Tabular](../data/tabular)
    - [Text extraction](../data/textractor)
- Image
    - [Caption](../image/caption)
    - [Objects](../image/objects)
- Text
    - [Extractive QA](../text/extractor)
    - [Labeling](../text/labels)
    - [Similarity](../text/similarity)
    - [Summary](../text/summary)
    - [Translation](../text/translation)
- Training
    - [HF ONNX](../train/hfonnx)
    - [ML ONNX](../train/mlonnx)
    - [Trainer](../train/trainer)

# Pipelines

txtai provides a generic pipeline processing framework with the only interface requirement being a `__call__` method. Pipelines are flexible and can
process various types of data. Pipelines can wrap machine learning models as well as other processes.

The following pipeline types are currently supported.

- Hugging Face pipelines
    - [Extractive QA](../extractor)
    - [Labeling](../labels)
    - [Similarity](../similarity)
    - [Summary](../summary)
- Hugging Face models
    - [Transcription](../transcription)
    - [Translation](../translation)
- Data processing calls
    - [Text extraction](../textractor)

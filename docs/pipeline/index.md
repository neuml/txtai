# Pipeline

![pipeline](../images/pipeline.png#only-light)
![pipeline](../images/pipeline-dark.png#only-dark)

txtai provides a generic pipeline processing framework with the only interface requirement being a `__call__` method. Pipelines are flexible and process various types of data. Pipelines can wrap machine learning models as well as other processes.

Pipelines are run with Python or configuration. Pipelines can be instantiated in [configuration](../api/configuration/#pipeline) using the lower case name of the pipeline. Configuration-driven pipelines are run with [workflows](../workflow/#configuration-driven-example) or the [API](../api#local-instance).

## List of pipelines

The following is a list of the current pipelines available in txtai. All pipelines use default models when otherwise not specified. See the [model guide](../models) for the current model recommendations. All pipelines are designed to work with local models via the [Transformers library](https://github.com/huggingface/transformers).

The `LLM` and `RAG` pipelines also have integrations for [llama.cpp](https://github.com/abetlen/llama-cpp-python) and [hosted API models via LiteLLM](https://github.com/BerriAI/litellm). The `LLM` pipeline can be prompted to accomplish many of the same tasks (i.e. summarization, translation, classification).

- Audio
    - [AudioMixer](audio/audiomixer)
    - [AudioStream](audio/audiostream)
    - [Microphone](audio/microphone)
    - [TextToAudio](audio/texttoaudio)
    - [TextToSpeech](audio/texttospeech)
    - [Transcription](audio/transcription)
- Data Processing
    - [FileToHTML](data/filetohtml)
    - [HTMLToMarkdown](data/htmltomd)
    - [Segmentation](data/segmentation)
    - [Tabular](data/tabular)
    - [Text extraction](data/textractor)
- Image
    - [Caption](image/caption)
    - [Image Hash](image/imagehash)
    - [Objects](image/objects)
- Text
    - [Entity](text/entity)
    - [Labeling](text/labels)
    - [LLM](text/llm)
    - [RAG](text/rag)
    - [Similarity](text/similarity)
    - [Summary](text/summary)
    - [Translation](text/translation)
- Training
    - [HF ONNX](train/hfonnx)
    - [ML ONNX](train/mlonnx)
    - [Trainer](train/trainer)

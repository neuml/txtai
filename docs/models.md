# Model guide

![models](images/models.png)

See the table below for the current recommended models. These models all allow commercial use and offer a blend of speed and performance.

| Component                                            | Model(s)                                                                 |
| ---------------------------------------------------- | ------------------------------------------------------------------------ |
| [Embeddings](../embeddings)                          | [all-MiniLM-L6-v2](https://hf.co/sentence-transformers/all-MiniLM-L6-v2) | 
| [Image Captions](./pipeline/image/caption.md)        | [BLIP](https://hf.co/Salesforce/blip-image-captioning-base)              |
| [Labels - Zero Shot](./pipeline/text/labels.md)      | [BART-Large-MNLI](https://hf.co/facebook/bart-large)                     |
| [Labels - Fixed](./pipeline/text/labels.md)          | Fine-tune with [training pipeline](./pipeline/train/trainer.md)          |
| [Large Language Model (LLM)](./pipeline/text/llm.md) | [Llama 3.1 Instruct](https://hf.co/meta-llama/Llama-3.1-8B-Instruct)     |
| [Summarization](./pipeline/text/summary.md)          | [DistilBART](https://hf.co/sshleifer/distilbart-cnn-12-6)                |
| [Text-to-Speech](./pipeline/audio/texttospeech.md)   | [ESPnet JETS](https://hf.co/NeuML/ljspeech-jets-onnx)                    |
| [Transcription](./pipeline/audio/transcription.md)   | [Whisper](https://hf.co/openai/whisper-base)                             | 
| [Translation](./pipeline/text/translation.md)        | [OPUS Model Series](https://hf.co/Helsinki-NLP)                          |

Models can be loaded as either a path from the Hugging Face Hub or a local directory. Model paths are optional, defaults are loaded when not specified. For tasks with no recommended model, txtai uses the default models as shown in the Hugging Face Tasks guide.

See the following links to learn more.

- [Hugging Face Tasks](https://hf.co/tasks)
- [Hugging Face Model Hub](https://hf.co/models)
- [MTEB Leaderboard](https://hf.co/spaces/mteb/leaderboard)
- [LMSYS LLM Leaderboard](https://chat.lmsys.org/?leaderboard)
- [Open LLM Leaderboard](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard)

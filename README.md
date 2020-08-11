# txtai: AI-powered search engine

txtai builds an AI-powered index over sets of text. txtai supports building text indices to perform similarity searches and create extractive question-answering based systems. 

![demo](https://raw.githubusercontent.com/neuml/txtai/master/demo.gif)

NeuML uses txtai and/or the concepts behind it to power all of our Natural Language Processing (NLP) applications. Example applications:

- [cord19q](https://github.com/neuml/cord19q) - COVID-19 literature analysis
- [paperai](https://github.com/neuml/paperai) - AI-powered literature discovery and review engine for medical/scientific papers
- [neuspo](https://neuspo.com) - a fact-driven, real-time sports event and news site
- [codequestion](https://github.com/neuml/codequestion) - Ask coding questions directly from the terminal

txtai is built on the following stack:

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [transformers](https://github.com/huggingface/transformers)
- [faiss](https://github.com/facebookresearch/faiss)
- Python 3.6+

## Installation
You can install txtai directly from GitHub using pip. Using a Python Virtual Environment is recommended.

    pip install git+https://github.com/neuml/txtai

Python 3.6+ is supported

### Notes for Windows
This project has dependencies that require compiling native code. Linux enviroments usually work without an issue. Windows requires the following extra steps.

- Install C++ Build Tools - https://visualstudio.microsoft.com/visual-cpp-build-tools/
- If PyTorch errors are encountered, run the following command before installing paperai. See [pytorch.org](https://pytorch.org) for more information.

    ```
    pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```

## Tutorials 
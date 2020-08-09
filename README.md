# txtai: AI-powered search engine

txtai build an AI-powered index over sets of text.

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

# pylint: disable = C0111
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    # Remove GitHub dark mode images
    DESCRIPTION = "".join([line for line in f if "gh-dark-mode-only" not in line])

# Required dependencies
install = ["faiss-cpu>=1.7.1.post2", "msgpack>=1.0.7", "torch>=1.12.1", "transformers>=4.45.0"]

# Required dependencies that are also transformers dependencies
install += ["huggingface-hub>=0.19.0", "numpy>=1.18.4", "pyyaml>=5.3", "regex>=2022.8.17"]

# Optional dependencies
extras = {}

# Development dependencies - not included in "all" install
extras["dev"] = [
    "black",
    "coverage",
    "coveralls",
    "httpx",
    "mkdocs-material",
    "mkdocs-redirects",
    "mkdocstrings[python]",
    "pre-commit",
    "pylint",
]

extras["agent"] = ["mcpadapt>=0.1.0", "smolagents>=1.17"]

extras["ann"] = [
    "annoy>=1.16.3",
    "hnswlib>=0.5.0",
    "pgvector>=0.3.0",
    "sqlalchemy>=2.0.20",
    "sqlite-vec>=0.1.1",
]

extras["api"] = [
    "aiohttp>=3.8.1",
    "fastapi>=0.94.0",
    "fastapi-mcp>=0.2.0",
    "httpx>=0.28.1",
    "pillow>=7.1.2",
    "python-multipart>=0.0.7",
    "uvicorn>=0.12.1",
]

extras["cloud"] = ["apache-libcloud>=3.3.1", "fasteners>=0.14.1"]

extras["console"] = ["rich>=12.0.1"]

extras["database"] = ["duckdb>=0.7.1", "pillow>=7.1.2", "sqlalchemy>=2.0.20"]

extras["graph"] = ["grand-cypher>=0.6.0", "grand-graph>=0.6.0", "networkx>=2.7.1", "sqlalchemy>=2.0.20"]

extras["model"] = ["onnx>=1.11.0", "onnxruntime>=1.11.0"]

extras["pipeline-audio"] = [
    "onnx>=1.11.0",
    "onnxruntime>=1.11.0",
    "scipy>=1.4.1",
    "sounddevice>=0.5.0",
    "soundfile>=0.10.3.post1",
    "ttstokenizer>=1.1.0",
    "webrtcvad-wheels>=2.0.14",
]

extras["pipeline-data"] = ["beautifulsoup4>=4.9.3", "chonkie>=1.0.2", "docling>=2.8.2", "nltk>=3.5", "pandas>=1.1.0", "tika>=1.24"]

extras["pipeline-image"] = ["imagehash>=4.2.1", "pillow>=7.1.2", "timm>=0.4.12"]

extras["pipeline-llm"] = ["litellm>=1.37.16", "llama-cpp-python>=0.2.75"]

extras["pipeline-text"] = ["gliner>=0.2.16", "sentencepiece>=0.1.91", "staticvectors>=0.2.0"]

extras["pipeline-train"] = [
    "accelerate>=0.26.0",
    "bitsandbytes>=0.42.0",
    "onnx>=1.11.0",
    "onnxmltools>=1.9.1",
    "onnxruntime>=1.11.0",
    "peft>=0.8.1",
    "skl2onnx>=1.9.1",
]

extras["pipeline"] = (
    extras["pipeline-audio"]
    + extras["pipeline-data"]
    + extras["pipeline-image"]
    + extras["pipeline-llm"]
    + extras["pipeline-text"]
    + extras["pipeline-train"]
)

extras["scoring"] = ["sqlalchemy>=2.0.20"]

extras["vectors"] = [
    "litellm>=1.37.16",
    "llama-cpp-python>=0.2.75",
    "model2vec>=0.3.0",
    "scikit-learn>=0.23.1",
    "sentence-transformers>=2.2.0",
    "skops>=0.9.0",
    "staticvectors>=0.2.0",
]

extras["workflow"] = [
    "apache-libcloud>=3.3.1",
    "croniter>=1.2.0",
    "openpyxl>=3.0.9",
    "pandas>=1.1.0",
    "pillow>=7.1.2",
    "requests>=2.26.0",
    "xmltodict>=0.12.0",
]

# Backwards-compatible combination of ann and vectors extra
extras["similarity"] = extras["ann"] + extras["vectors"]

extras["all"] = (
    extras["agent"]
    + extras["api"]
    + extras["cloud"]
    + extras["console"]
    + extras["database"]
    + extras["graph"]
    + extras["model"]
    + extras["pipeline"]
    + extras["scoring"]
    + extras["similarity"]
    + extras["workflow"]
)

setup(
    name="txtai",
    version="8.7.0",
    author="NeuML",
    description="All-in-one open-source AI framework for semantic search, LLM orchestration and language model workflows",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neuml/txtai",
    project_urls={
        "Documentation": "https://github.com/neuml/txtai",
        "Issue Tracker": "https://github.com/neuml/txtai/issues",
        "Source Code": "https://github.com/neuml/txtai",
    },
    license="Apache 2.0: http://www.apache.org/licenses/LICENSE-2.0",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    keywords="search embedding machine-learning nlp",
    python_requires=">=3.10",
    install_requires=install,
    extras_require=extras,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Utilities",
    ],
)

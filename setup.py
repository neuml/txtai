# pylint: disable = C0111
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    # Remove GitHub dark mode images
    DESCRIPTION = "".join([line for line in f if "gh-dark-mode-only" not in line])

# Required dependencies
install = ["faiss-cpu>=1.7.1.post2", "numpy>=1.18.4", "pyyaml>=5.3", "torch>=1.6.0", "transformers>=4.12.3", "tokenizers>=0.10.3"]

# Optional dependencies
extras = {}

extras["dev"] = ["black", "coverage", "pre-commit", "pylint"]

extras["api"] = [
    "aiohttp>=3.8.1",
    "fastapi>=0.61.1",
    "uvicorn>=0.12.1",
]

extras["cloud"] = ["apache-libcloud>=3.3.1"]

extras["database"] = ["pillow>=9.0.1"]

extras["model"] = ["onnxruntime>=1.8.1"]

extras["pipeline"] = [
    "beautifulsoup4>=4.9.3",
    "fasttext>=0.9.2",
    "imagehash>=4.2.1",
    "nltk>=3.5",
    "onnx>=1.10.1",
    "onnxmltools>=1.9.1",
    "onnxruntime>=1.8.1",
    "pandas>=1.1.0",
    "pillow>=9.0.1",
    "sentencepiece>=0.1.91",
    "soundfile>=0.10.3.post1",
    "tika>=1.24",
    "timm>=0.4.12",
]

extras["similarity"] = [
    "annoy>=1.16.3",
    "fasttext>=0.9.2",
    "hnswlib>=0.5.0",
    "pymagnitude-lite>=0.1.43",
    "scikit-learn>=0.23.1",
    "sentence-transformers>=2.1.0",
]

extras["workflow"] = [
    "apache-libcloud>=3.3.1",
    "croniter>=1.2.0",
    "openpyxl>=3.0.9",
    "pandas>=1.1.0",
    "pillow>=9.0.1",
    "requests>=2.26.0",
    "xmltodict>=0.12.0",
]

extras["all"] = (
    extras["api"] + extras["cloud"] + extras["database"] + extras["model"] + extras["pipeline"] + extras["similarity"] + extras["workflow"]
)

setup(
    name="txtai",
    version="4.3.1",
    author="NeuML",
    description="Build AI-powered semantic search applications",
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
    python_requires=">=3.7",
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

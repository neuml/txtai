# pylint: disable = C0111
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    DESCRIPTION = f.read()

setup(name="txtai",
      version="1.3.0",
      author="NeuML",
      description="AI-powered search engine",
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
      python_requires=">=3.6",
      install_requires=[
          "annoy>=1.16.3",
          "faiss-cpu>=1.6.3; os_name != 'nt'",
          "fastapi>=0.61.1",
          "fasttext>=0.9.2",
          "hnswlib>=0.4.0",
          "nltk>=3.5",
          "numpy>=1.18.4",
          "pymagnitude-lite>=0.1.43",
          "PyYAML>=5.3",
          "regex>=2020.5.14",
          "scikit-learn>=0.23.1",
          "torch>=1.4.0",
          "tqdm>=4.46.0",
          "sentence-transformers>=0.3.6",
          "transformers>=3.1.0",
          "uvicorn>=0.12.1"
      ],
      classifiers=[
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Software Development",
          "Topic :: Text Processing :: Indexing",
          "Topic :: Utilities"
      ])

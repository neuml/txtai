"""
FastAPI application module
"""

import os

import yaml

from fastapi import FastAPI

from .base import API
from .factory import Factory

# pylint: disable=R0401
from .routers import embeddings, extractor, labels, segmentation, similarity, summary, textractor, transcription, translation, workflow

# API instance
app = FastAPI()

# Global API instance
INSTANCE = None


def get():
    """
    Returns global API instance.

    Returns:
        API instance
    """

    # pylint: disable=W0603
    global INSTANCE
    return INSTANCE


@app.on_event("startup")
def start():
    """
    FastAPI startup event. Loads API instance.
    """

    # pylint: disable=W0603
    global INSTANCE

    # Load YAML settings
    with open(os.getenv("CONFIG"), "r") as f:
        # Read configuration
        config = yaml.safe_load(f)

    # Instantiate API instance
    api = os.getenv("API_CLASS")
    INSTANCE = Factory.create(config, api) if api else API(config)

    # Router definitions
    routers = [
        ("embeddings", embeddings.router),
        ("extractor", extractor.router),
        ("labels", labels.router),
        ("segmentation", segmentation.router),
        ("similarity", similarity.router),
        ("summary", summary.router),
        ("textractor", textractor.router),
        ("transcription", transcription.router),
        ("translation", translation.router),
        ("workflow", workflow.router),
    ]

    # Conditionally add routes based on configuration
    for name, router in routers:
        if name in config:
            app.include_router(router)

    # Special case for embeddings clusters
    if "cluster" in config and "embeddings" not in config:
        app.include_router(embeddings.router)

    # Special case to add similarity instance for embeddings
    if "embeddings" in config and "similarity" not in config:
        app.include_router(similarity.router)

    # Execute extensions if present
    extensions = os.getenv("EXTENSIONS")
    if extensions:
        for extension in extensions.split(","):
            # Create instance and execute extension
            extension = Factory.get(extension.strip())()
            extension(app)

"""
FastAPI application module
"""

import inspect
import os

from fastapi import FastAPI

from .base import API
from .factory import Factory
from .routers import embeddings, similarity

from . import routers

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

    return INSTANCE


def apirouters():
    """
    Lists available APIRouters.

    Returns:
        list of (router name, router)
    """

    available = []
    for name, rclass in inspect.getmembers(routers, inspect.ismodule):
        if hasattr(rclass, "router"):
            available.append((name.lower(), rclass.router))

    return available


@app.on_event("startup")
def start():
    """
    FastAPI startup event. Loads API instance.
    """

    # pylint: disable=W0603
    global INSTANCE

    # Load YAML settings
    config = API.read(os.getenv("CONFIG"))

    # Instantiate API instance
    api = os.getenv("API_CLASS")
    INSTANCE = Factory.create(config, api) if api else API(config)

    # Conditionally add routes based on configuration
    for name, router in apirouters():
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

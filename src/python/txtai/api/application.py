"""
FastAPI application module
"""

import inspect
import os
import sys

from fastapi import APIRouter, FastAPI

from .base import API
from .factory import Factory

from ..app import Application

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
        {router name: router}
    """

    # Get handle to api module
    api = sys.modules[".".join(__name__.split(".")[:-1])]

    available = {}
    for name, rclass in inspect.getmembers(api, inspect.ismodule):
        if hasattr(rclass, "router") and isinstance(rclass.router, APIRouter):
            available[name.lower()] = rclass.router

    return available


@app.on_event("startup")
def start():
    """
    FastAPI startup event. Loads API instance.
    """

    # pylint: disable=W0603
    global INSTANCE

    # Load YAML settings
    config = Application.read(os.getenv("CONFIG"))

    # Instantiate API instance
    api = os.getenv("API_CLASS")
    INSTANCE = Factory.create(config, api) if api else API(config)

    # Get all known routers
    routers = apirouters()

    # Conditionally add routes based on configuration
    for name, router in routers.items():
        if name in config:
            app.include_router(router)

    # Special case for embeddings clusters
    if "cluster" in config and "embeddings" not in config:
        app.include_router(routers["embeddings"])

    # Special case to add similarity instance for embeddings
    if "embeddings" in config and "similarity" not in config:
        app.include_router(routers["similarity"])

    # Execute extensions if present
    extensions = os.getenv("EXTENSIONS")
    if extensions:
        for extension in extensions.split(","):
            # Create instance and execute extension
            extension = Factory.get(extension.strip())()
            extension(app)

"""
FastAPI application module
"""

import inspect
import os
import sys

from fastapi import APIRouter, Depends, FastAPI
from fastapi_mcp import FastApiMCP
from httpx import AsyncClient

from .authorization import Authorization
from .base import API
from .factory import APIFactory

from ..app import Application


def get():
    """
    Returns global API instance.

    Returns:
        API instance
    """

    return INSTANCE


def create():
    """
    Creates a FastAPI instance.
    """

    # Application dependencies
    dependencies = []

    # Default implementation of token authorization
    token = os.environ.get("TOKEN")
    if token:
        dependencies.append(Depends(Authorization(token)))

    # Add custom dependencies
    deps = os.environ.get("DEPENDENCIES")
    if deps:
        for dep in deps.split(","):
            # Create and add dependency
            dep = APIFactory.get(dep.strip())()
            dependencies.append(Depends(dep))

    # Create FastAPI application
    return FastAPI(lifespan=lifespan, dependencies=dependencies if dependencies else None)


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


def lifespan(application):
    """
    FastAPI lifespan event handler.

    Args:
        application: FastAPI application to initialize
    """

    # pylint: disable=W0603
    global INSTANCE

    # Load YAML settings
    config = Application.read(os.environ.get("CONFIG"))

    # Instantiate API instance
    api = os.environ.get("API_CLASS")
    INSTANCE = APIFactory.create(config, api) if api else API(config)

    # Get all known routers
    routers = apirouters()

    # Conditionally add routes based on configuration
    for name, router in routers.items():
        if name in config:
            application.include_router(router)

    # Special case for embeddings clusters
    if "cluster" in config and "embeddings" not in config:
        application.include_router(routers["embeddings"])

    # Special case to add similarity instance for embeddings
    if "embeddings" in config and "similarity" not in config:
        application.include_router(routers["similarity"])

    # Execute extensions if present
    extensions = os.environ.get("EXTENSIONS")
    if extensions:
        for extension in extensions.split(","):
            # Create instance and execute extension
            extension = APIFactory.get(extension.strip())()
            extension(application)

    # Add Model Context Protocol (MCP) service, if applicable
    if config.get("mcp"):
        mcp = FastApiMCP(application, http_client=AsyncClient(timeout=100))
        mcp.mount()

    yield


def start():
    """
    Runs application lifespan handler.
    """

    list(lifespan(app))


# FastAPI instance txtai API instances
app, INSTANCE = create(), None

"""
Pipeline factory module
"""

import inspect
import sys
import types

from .base import Pipeline


class PipelineFactory:
    """
    Pipeline factory. Creates new Pipeline instances.
    """

    @staticmethod
    def get(pipeline):
        """
        Gets a new instance of pipeline class.

        Args:
            pclass: Pipeline instance class

        Returns:
            Pipeline class
        """

        # Local pipeline if no package
        if "." not in pipeline:
            return PipelineFactory.list()[pipeline]

        # Attempt to load custom pipeline
        parts = pipeline.split(".")
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

        return m

    @staticmethod
    def create(config, pipeline):
        """
        Creates a new Pipeline instance.

        Args:
            config: Pipeline configuration
            pipeline: Pipeline instance class

        Returns:
            Pipeline
        """

        # Resolve pipeline
        pipeline = PipelineFactory.get(pipeline)

        # Return functions directly, otherwise create pipeline instance
        return pipeline if isinstance(pipeline, types.FunctionType) else pipeline(**config)

    @staticmethod
    def list():
        """
        Lists callable pipelines.

        Returns:
            {short name: pipeline class}
        """

        pipelines = {}

        # Get handle to pipeline module
        pipeline = sys.modules[".".join(__name__.split(".")[:-1])]

        # Get list of callable pipelines
        for x in inspect.getmembers(pipeline, inspect.isclass):
            if issubclass(x[1], Pipeline) and [y for y, _ in inspect.getmembers(x[1], inspect.isfunction) if y == "__call__"]:
                # short name: pipeline class
                pipelines[x[0].lower()] = x[1]

        return pipelines

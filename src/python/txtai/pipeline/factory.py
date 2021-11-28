"""
Pipeline factory module
"""

import inspect
import sys

from .base import Pipeline


class PipelineFactory:
    """
    Pipeline factory. Creates new Pipeline instances.
    """

    @staticmethod
    def get(pclass):
        """
        Gets a new instance of pipeline class.

        Args:
            pclass: Pipeline instance class

        Returns:
            Pipeline class
        """

        # Local pipeline if no package
        if "." not in pclass:
            return PipelineFactory.list()[pclass]

        # Attempt to load custom pipeline
        parts = pclass.split(".")
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

        return m

    @staticmethod
    def create(config, pclass):
        """
        Creates a new Pipeline instance.

        Args:
            config: Pipeline configuration
            pclass: Pipeline instance class

        Returns:
            Pipeline
        """

        # Get Pipeline instance
        return PipelineFactory.get(pclass)(**config)

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

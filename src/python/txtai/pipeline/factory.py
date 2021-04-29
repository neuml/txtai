"""
Pipeline factory module
"""


class PipelineFactory:
    """
    Pipeline factory. Creates new Pipeline instances.
    """

    @staticmethod
    def get(pipeline):
        """
        Gets a new instance of pipeline class.

        Args:
            pipeline: Pipeline instance class

        Returns:
            Pipeline class
        """

        # Local pipeline if no package
        if "." not in pipeline:
            # Get parent package
            pipeline = ".".join(__name__.split(".")[:-1]) + "." + pipeline.capitalize()

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

        # Get Pipeline instance
        return PipelineFactory.get(pipeline)(**config)

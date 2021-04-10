"""
Extension module
"""


class Extension:
    """
    Defines an API extension. API extensions can expose custom pipelines or other custom logic.
    """

    def __call__(self, app):
        """
        Hook to register custom routing logic and/or modify the FastAPI instance.

        Args:
            app: FastAPI application instance
        """

        return

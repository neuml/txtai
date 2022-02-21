"""
Lambda handler for txtai workflows
"""

from txtai.api import API

APP = None

# pylint: disable=W0603,W0613
def handler(event, context):
    """
    Runs a workflow using input event parameters.

    Args:
        event: input event
        context: input context

    Returns:
        Workflow results
    """

    # Create (or get) global app instance
    global APP
    APP = APP if APP else API("config.yml")

    # Run workflow and return results
    return list(APP.workflow(event["name"], event["elements"]))

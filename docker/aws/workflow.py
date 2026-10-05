"""
Lambda handler for txtai workflows
"""

import json

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

    # Get parameters from event body
    event = json.loads(event["body"])

    # Run workflow and return results
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": list(APP.workflow(event["name"], event["elements"]))}

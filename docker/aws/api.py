"""
Lambda handler for a txtai API instance
"""

from mangum import Mangum

from txtai.api import app, start

# pylint: disable=C0103
# Create FastAPI application instance wrapped by Mangum
handler = None
if not handler:
    # Start application
    start()

    # Create handler
    handler = Mangum(app, lifespan="off")

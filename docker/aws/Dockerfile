# Set base image
ARG BASE_IMAGE=neuml/txtai-cpu
FROM $BASE_IMAGE

# Application script to copy into image
ARG APP=api.py

# Install Lambda Runtime Interface Client and Mangum ASGI bindings
RUN pip install awslambdaric mangum

# Copy configuration
COPY config.yml .

# Run local API instance to cache models in container
RUN python -c "from txtai.api import API; API('config.yml', False)"

# Copy application
COPY $APP ./app.py

# Start runtime client using default application handler
ENV CONFIG "config.yml"
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["app.handler"]

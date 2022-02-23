# Set base image
ARG BASE_IMAGE=neuml/txtai-cpu
FROM $BASE_IMAGE

# Copy configuration
COPY config.yml .

# Run local API instance to cache models in container
RUN python -c "from txtai.api import API; API('config.yml', False)"

# Start application and wait for completion. Scheduled workflows can run indefinitely. 
ENTRYPOINT ["python", "-c", "from txtai.api import API; API('config.yml').wait()"]

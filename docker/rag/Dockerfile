# Set base image
ARG BASE_IMAGE=neuml/txtai-cpu
FROM $BASE_IMAGE

# Copy RAG application
COPY rag.py .

# Install required dependencies
RUN python -m pip install --no-cache-dir streamlit autoawq

# Start streamlit application
ENTRYPOINT ["streamlit", "run", "rag.py"]

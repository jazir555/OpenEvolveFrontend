# Dockerfile for Curie-GlobalChem Integration Adapter
#
# Following CLAUDE.md principles:
# - Containerized deployment
# - Immutable infrastructure
# - Explicit configuration
# - Reproducible builds

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the adapter source code
COPY ./src/ ./src/
COPY config.json ./

# Create probes directory and copy probe script
RUN mkdir -p probes
COPY probes/integration_probe.py ./probes/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python probes/integration_probe.py

# Default command
CMD ["python", "-c", "from src.curie_globalchem_adapter import CurieGlobalChemAdapter; adapter = CurieGlobalChemAdapter(); print('Adapter initialized successfully')"]
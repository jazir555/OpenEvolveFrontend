# Dockerfile for PAMI - Research Quest - Curie-GlobalChem Integration Adapter
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

# Create probes directory
RUN mkdir -p probes

# Set environment variables
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from src.pami_research_quest_curie_globalchem_adapter import PAMIResearchQuestCurieGlobalChemAdapter; print('Health check passed')"

# Default command
CMD ["python", "-c", "import asyncio; from src.pami_research_quest_curie_globalchem_adapter import main; asyncio.run(main())"]
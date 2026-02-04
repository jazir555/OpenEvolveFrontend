# RESE Deep Exploration Engine Adapter Dockerfile
# Follows CLAUDE.md principles: isolated container, no file sharing with core

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables (Law of Configuration Explicitness)
ENV EXPLORATION_DEPTH=10 \
    MCTS_ITERATIONS=1000 \
    MCTS_EXPLORATION_CONSTANT=1.414 \
    CONVERGENCE_THRESHOLD=0.001 \
    EXPLORATION_TIMEOUT_MS=10000 \
    MAX_HYPOTHESES=100 \
    PATTERN_RECOGNITION_THRESHOLD=0.7 \
    BEAM_WIDTH=10 \
    TEMPERATURE=1.0 \
    POPULATION_SIZE=50 \
    MUTATION_RATE=0.1 \
    DLQ_MAX_SIZE=1000 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy adapter code (isolated from core projects)
COPY src/ ./src/
COPY ../schemas/ ../schemas/
COPY ../lib/ ../lib/

# Create logs directory
RUN mkdir -p /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.dee_adapter import DEEAdapter; adapter = DEEAdapter(); print(adapter.get_health())"

# Default command
CMD ["python", "-u", "src/dee_adapter.py"]

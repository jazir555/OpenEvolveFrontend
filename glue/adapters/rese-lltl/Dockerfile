# RESE LLTL Adapter Dockerfile
#
# This Dockerfile creates a containerized environment for the
# Logic-to-Loss Translation Layer adapter.
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via env vars
# - Law of Runtime Truth: Verify before using
# - Circuit Breaker: Handle failures gracefully
#
# Author: RESE Team
# Created: 2026-02-04

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables (Law of Configuration Explicitness)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLTL_ENCODING_DIM=128 \
    LLTL_TIMEOUT_MS=3000 \
    LLTL_LEARNING_RATE=0.001 \
    LLTL_CACHE_SIZE=1000 \
    LLTL_DEFAULT_LOSS_TYPE=mse \
    LLTL_COMBINATION_STRATEGY=weighted_sum \
    LLTL_NORMALIZE_WEIGHTS=true \
    LLTL_CONTRADICTION_THRESHOLD=0.8 \
    LLTL_MAX_CONTRADICTIONS=1000

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy project structure
# Note: Following Law of the Air Gap - no imports from core-projects
COPY glue/lib/rese_lltl.py /app/glue/lib/
COPY glue/adapters/rese-lltl/src/lltl_adapter.py /app/glue/adapters/rese-lltl/src/
COPY glue/adapters/rese-lltl/probes/check_lltl.sh /app/glue/adapters/rese-lltl/probes/

# Create __init__ files for proper Python package structure
RUN touch /app/glue/__init__.py && \
    touch /app/glue/lib/__init__.py && \
    touch /app/glue/adapters/__init__.py && \
    touch /app/glue/adapters/rese-lltl/__init__.py && \
    touch /app/glue/adapters/rese-lltl/src/__init__.py

# Make probe script executable
RUN chmod +x /app/glue/adapters/rese-lltl/probes/check_lltl.sh

# Add to Python path
ENV PYTHONPATH="/app/glue/lib:/app/glue/adapters/rese-lltl/src:${PYTHONPATH}"

# Health check (Law of Runtime Truth)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from lltl_adapter import is_available; exit(0 if is_available() else 1)" || exit 1

# Run probe script on startup to verify installation
RUN echo "#!/bin/bash" > /app/entrypoint.sh && \
    echo "echo 'Running LLTL health check...'" >> /app/entrypoint.sh && \
    echo "cd /app/glue/adapters/rese-lltl" >> /app/entrypoint.sh && \
    echo "bash probes/check_lltl.sh || exit 1" >> /app/entrypoint.sh && \
    echo "echo 'LLTL is ready!'" >> /app/entrypoint.sh && \
    echo "exec \"\$@\"" >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command: Run a simple test
CMD ["python", "-c", "
import sys;
sys.path.insert(0, '/app/glue/lib');
sys.path.insert(0, '/app/glue/adapters/rese-lltl/src');
from lltl_adapter import LLTLAdapter, is_available;
print(f'LLTL Available: {is_available()}');
adapter = LLTLAdapter();
healthy, msg = adapter.health_check();
print(f'Health: {healthy} - {msg}');
print('LLTL adapter is running!');
"]

# Labels for metadata
LABEL maintainer="RESE Team" \
      description="RESE Logic-to-Loss Translation Layer Adapter" \
      version="1.0.0" \
      created="2026-02-04"

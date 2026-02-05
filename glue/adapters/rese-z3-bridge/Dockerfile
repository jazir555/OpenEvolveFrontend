# RESE-Z3 Bridge Adapter Dockerfile
#
# Following CLAUDE.md principles:
# - Law of the "Air Gap": No imports from core-projects
# - Law of Configuration Explicitness: All config via environment
# - Minimal base image for security
#
# Author: RESE Team
# Created: 2026-02-04

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy bridge source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY probes/ ./probes/

# Create non-root user for security
RUN useradd -m -u 1000 rese && \
    chown -R rese:rese /app

# Switch to non-root user
USER rese

# Environment variables (Law of Configuration Explicitness)
ENV Z3_BASE_URL=http://z3-core:8000 \
    Z3_TIMEOUT_MS=30000 \
    Z3_CIRCUIT_BREAKER_THRESHOLD=5 \
    Z3_CIRCUIT_BREAKER_TIMEOUT_MS=60000 \
    Z3_MAX_RETRIES=3 \
    Z3_RETRY_BACKOFF_MS=1000 \
    Z3_ENABLE_CACHE=true \
    Z3_CACHE_TTL_MS=300000 \
    Z3_ENABLE_MONITORING=true \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'src'); from rese_z3_bridge import RESEZ3Bridge; bridge = RESEZ3Bridge(); health = bridge.get_health(); exit(0 if health['status'] == 'healthy' else 1)" || exit 1

# Expose metrics port (if monitoring enabled)
EXPOSE 9090

# Default command: Run probe to verify then start
CMD ["sh", "-c", "python probes/check_z3_bridge.py && tail -f /dev/null"]

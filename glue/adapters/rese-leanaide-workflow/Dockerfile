# Dockerfile for LeanAide-RESE Workflow Adapter
#
# This Dockerfile creates a containerized adapter that integrates
# LeanAide's AI-powered theorem proving with RESE's 4-phase pipeline.
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via env vars
# - Law of Idempotency: Safe to restart
# - Minimal base image for security
#
# Author: OpenEvolve
# Version: 1.0.0

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY tests/ /app/tests/

# Create necessary directories
RUN mkdir -p /app/logs \
    /app/.leanaide_cache \
    /app/.lean_cache

# Set permissions
RUN chmod +x /app/src/*.py

# Create non-root user for security
RUN useradd -m -u 1000 reseuser && \
    chown -R reseuser:reseuser /app

USER reseuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "-m", "src.leanaide_rese_workflow"]

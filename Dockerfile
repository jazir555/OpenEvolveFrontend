# Z3-LeanAIDE-OpenEvolve-BubbleLabs Integration
# Multi-stage Docker build for production deployment

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libz3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="OpenEvolve"
LABEL version="2.0.0"
LABEL description="Z3-LeanAIDE-OpenEvolve-BubbleLabs Integration"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libz3-4 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r z3user && useradd -r -g z3user z3user

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/z3user/.local

# Copy application code
COPY z3prover_integration.py .
COPY z3prover_advanced.py .
COPY z3_leanaide_bridge.py .
COPY z3_leanaide_openevolve_integration.py .
COPY z3_leanaide_bubblelabs_ui.py .
COPY z3_bubblelabs_advanced_ui.py .
COPY z3_mcp_tools.py .
COPY z3_crewai_bridge.py .
COPY z3_result_cache.py .
COPY z3_performance_monitor.py .
COPY z3_knowledge_extraction.py .
COPY z3_config_manager.py .
COPY z3_database_models.py .
COPY z3_api_server.py .
COPY z3_cli.py .

# Copy configuration
COPY z3_config.yaml .

# Create data directories
RUN mkdir -p /app/data /app/logs /app/profiles && \
    chown -R z3user:z3user /app

# Set environment
ENV PATH=/home/z3user/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV Z3_CONFIG_PATH=/app/z3_config.yaml

# Switch to non-root user
USER z3user

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8765/health')" || exit 1

# Default command
CMD ["python", "-m", "z3_api_server"]

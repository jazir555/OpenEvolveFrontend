# =============================================================================
# Adaptive MDAP/MAKER Adapter Dockerfile
# =============================================================================
# Federation Constitution Compliant Multi-Stage Container Build
#
# Build:
#   docker build -t adaptive-mdap-adapter:1.0.0 .
#
# Run:
#   docker run --rm \
#     -e ADAPTIVE_MDAP_TIMEOUT_MS=5000 \
#     -e ADAPTIVE_MDAP_LOG_LEVEL=INFO \
#     adaptive-mdap-adapter:1.0.0
#
# Required Environment Variables:
#   - ADAPTIVE_MDAP_TIMEOUT_MS: Request timeout in milliseconds
#
# Optional Environment Variables:
#   - ADAPTIVE_MDAP_MAX_RETRIES: Maximum retry attempts (default: 3)
#   - ADAPTIVE_MDAP_RETRY_DELAY_MS: Initial retry delay (default: 100)
#   - ADAPTIVE_MDAP_CIRCUIT_BREAKER_THRESHOLD: Failures before opening (default: 5)
#   - ADAPTIVE_MDAP_LOG_LEVEL: Logging level (default: INFO)
# =============================================================================

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies to a temporary location
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

# Set metadata labels
LABEL maintainer="OpenEvolve Team"
LABEL version="${VERSION}"
LABEL description="Adaptive MDAP/MAKER Adapter - Anti-Corruption Layer"
LABEL build-date="${BUILD_DATE}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security (Federation Constitution compliance)
RUN groupadd -r adaptive && useradd -r -g adaptive adaptive

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY probes/ /app/probes/

# Create logs directory
RUN mkdir -p /app/logs && chown -R adaptive:adaptive /app

# Switch to non-root user
USER adaptive

# Health check (Federation Constitution - Runtime Truth)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from adaptive_mdap_adapter import get_adapter; adapter = get_adapter(); health = adapter.health_check(); sys.exit(0 if health['status'] == 'healthy' else 1)" || exit 1

# Default command: Run contract tests on startup
# If tests pass, start the health check loop
CMD python -m pytest /app/tests/contract.test.py -v --tb=short && \
    echo "Contract tests passed. Adapter ready." && \
    while true; do \
        sleep 60; \
        python -c "import sys; sys.path.insert(0, '/app/src'); from adaptive_mdap_adapter import get_adapter; adapter = get_adapter(); health = adapter.health_check(); print(f'Health: {health}')" ; \
    done

# =============================================================================
# Alternative: Run probes instead of contract tests
# =============================================================================
# To run probes on startup instead:
# CMD /app/probes/check_adaptive_mdap_api.sh && \
#     /app/probes/check_maker_api.sh && \
#     /app/probes/check_integration.sh && \
#     echo "All probes passed. Adapter ready." && \
#     tail -f /dev/null
# =============================================================================

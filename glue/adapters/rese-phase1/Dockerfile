# RESE Phase I Adapter Dockerfile
#
# Following CLAUDE.md principles:
# - Law of the "Air Gap": Isolated from core-projects
# - Law of Configuration Explicitness: All config via env vars
# - Circuit Breaker Pattern: Built-in failure detection
# - Structured Logging: JSON output
# - Timeout Enforcement: All operations have timeouts

FROM python:3.11-slim

# Metadata
LABEL maintainer="RESE Integration Team"
LABEL description="RESE Phase I: Epistemic Audit Adapter"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Phase I adapter code
COPY src/ /app/src/
COPY probes/ /app/probes/

# Install Python dependencies (if any)
# Currently using only standard library
# COPY requirements.txt /app/
# RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 rese && \
    chown -R rese:rese /app

USER rese

# Environment variables (Law of Configuration Explicitness)
# All these must be set at runtime or have safe defaults
ENV PHASE1_TIMEOUT_MS=15000 \
    PHASE1_CONSTRAINT_TIMEOUT_MS=5000 \
    PHASE1_ASSUMPTION_TIMEOUT_MS=5000 \
    PHASE1_CONTRADICTION_TIMEOUT_MS=10000 \
    PHASE1_FALSIFICATION_TIMEOUT_MS=5000 \
    PHASE1_MAX_ASSUMPTIONS=100 \
    PHASE1_MAX_CONSTRAINTS=1000 \
    PHASE1_MAX_CONTRADICTIONS=100 \
    PHASE1_MAX_FALSIFICATION_ATTEMPTS=50 \
    PHASE1_CIRCUIT_BREAKER_THRESHOLD=5 \
    PHASE1_CIRCUIT_BREAKER_TIMEOUT_MS=60000 \
    PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.3 \
    PHASE1_MIN_ROBUSTNESS_SCORE=0.5 \
    PHASE1_ENABLE_TACIT_MINING=true \
    PHASE1_ENABLE_LEAN4=false \
    PHASE1_ENABLE_RED_TEAM=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 /app/src/phase1_adapter.py health || exit 1

# Default command
CMD ["python3", "/app/src/phase1_executor.py", "--help"]

# Expose port for health checks (if running as service)
EXPOSE 8080

# Volumes for logs and data
VOLUME ["/app/logs", "/app/data"]

# Documentation
RUN echo "RESE Phase I: Epistemic Audit Adapter" > /README.txt && \
    echo "=====================================" >> /README.txt && \
    echo "" >> /README.txt && \
    echo "Environment Variables (required):" >> /README.txt && \
    echo "  PHASE1_TIMEOUT_MS              - Overall timeout (default: 15000)" >> /README.txt && \
    echo "  PHASE1_MAX_ASSUMPTIONS         - Max assumptions to mine (default: 100)" >> /README.txt && \
    echo "  PHASE1_CIRCUIT_BREAKER_THRESHOLD - Circuit breaker threshold (default: 5)" >> /README.txt && \
    echo "" >> /README.txt && \
    echo "Usage:" >> /README.txt && \
    echo "  docker run -e PHASE1_TIMEOUT_MS=20000 \\ " >> /README.txt && \
    echo "    -v \$(pwd)/data:/app/data \\ " >> /README.txt && \
    echo "    rese-phase1:latest \\ " >> /README.txt && \
    echo "    python3 /app/src/phase1_executor.py \\ " >> /README.txt && \
    echo "    --problem 'Problem description' \\ " >> /README.txt && \
    echo "    --patterns '[{\"pattern_description\": \"Test\", \"failure_rate\": 0.5, \"data_points\": 10}]'" >> /README.txt

# RESE Phase IV: Architecture Assembly Adapter
# Dockerfile following CLAUDE.md principles
#
# Law of Configuration Explicitness:
# All configuration via environment variables
# Crash immediately if required vars are missing

FROM python:3.11-slim

LABEL maintainer="RESE Development Team"
LABEL description="RESE Phase IV: Architecture Assembly Adapter"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy adapter code
COPY src/ /app/src/
COPY probes/ /app/probes/

# Create schemas directory link
RUN mkdir -p /app/schemas

# Set Python path
ENV PYTHONPATH=/app:/app/src:/app/schemas

# Create non-root user
RUN useradd -m -u 1000 rese && \
    chown -R rese:rese /app

USER rese

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "from src.adapter import Phase4Adapter; a = Phase4Adapter(); print(a.health_check()['status'])" || exit 1

# Default command: Run probe to verify functionality
CMD ["python3", "-c", "from src.adapter import Phase4Adapter; print('Phase IV Adapter ready')"]

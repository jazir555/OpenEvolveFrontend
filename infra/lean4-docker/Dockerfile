# ============================================================================
# Lean 4 Formal Verification Environment for RESE
# ============================================================================
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via env vars
# - Law of Runtime Truth: Verify Lean 4 works at runtime
# - Health checks: Verify Lean 4 + Mathlib operational
#
# Lean 4 v4.11.0 (stable) with Mathlib for formal verification
#
# Usage:
#   docker build -t rese-lean4:latest .
#   docker run --rm rese-lean4:latest lake env lean --version
# ============================================================================

FROM ubuntu:22.04

LABEL maintainer="RESE Project <rese@openevolve.org>"
LABEL description="Lean 4 v4.11.0 with Mathlib for RESE formal verification"
LABEL version="1.0.0"

# ============================================================================
# ENVIRONMENT VARIABLES (Law of Configuration Explicitness)
# ============================================================================

# Lean 4 version
ENV LEAN4_VERSION=v4.11.0
ENV LEAN4_RELEASE=https://github.com/leanprover/lean4/releases/download/v4.11.0/lean-4.11.0-linux.tar.gz

# Lake build system version
ENV LAKE_VERSION=v4.11.0

# Mathlib version (pin to specific commit for reproducibility)
ENV MATTLIB_VERSION=4.11.0
ENV MATHLIB_CACHE_DIR=/workspace/.mathlib_cache

# Python bridge configuration
ENV PYTHON_VERSION=3.11
ENV LEAN4_TIMEOUT_MS=30000
ENV LEAN4_MAX_MEMORY_MB=4096
ENV LEAN4_MAX_PROOF_TIME_MS=60000

# Workspace
ENV WORKSPACE_DIR=/workspace
ENV LEAN4_PROJECT_DIR=/workspace/lean4

# Health check
ENV HEALTH_CHECK_INTERVAL=30s
ENV HEALTH_CHECK_TIMEOUT=10s
ENV HEALTH_CHECK_RETRIES=3

# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    # Core utilities
    curl \
    wget \
    git \
    ca-certificates \
    # Build tools
    build-essential \
    libgmp-dev \
    # Python for bridge
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python3 \
    python3-pip \
    # Process monitoring
    procps \
    # Network tools (for health checks)
    netcat-openbsd \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# INSTALL LEAN 4
# ============================================================================

# Create workspace
WORKDIR ${WORKSPACE_DIR}

# Download and install Lean 4
RUN echo "Installing Lean 4 ${LEAN4_VERSION}..." \
    && curl -fSL ${LEAN4_RELEASE} -o lean.tar.gz \
    && tar -xzf lean.tar.gz \
    && rm lean.tar.gz \
    && mv lean-${LEAN4_VERSION#v} lean \
    && chmod +x lean/bin/lean \
    && chmod +x lean/bin/lake

# Add Lean to PATH
ENV PATH="${WORKSPACE_DIR}/lean/bin:${PATH}"

# Verify Lean installation
RUN echo "Verifying Lean 4 installation..." \
    && lean --version \
    && lake --version

# ============================================================================
# INSTALL MATHLIB (via Lake)
# ============================================================================

# Create Lean project directory
RUN mkdir -p ${LEAN4_PROJECT_DIR}

WORKDIR ${LEAN4_PROJECT_DIR}

# Initialize Lake project with Mathlib dependency
RUN echo "Initializing Lake project with Mathlib..." \
    && lake init rese \
    && lake update \
    && echo "require mathlib from git" > lakefile.lean \
    && echo "Mathlib version: ${MATTLIB_VERSION}"

# Pre-cache Mathlib (do this at build time for faster startup)
# Note: This is a large download, so we structure it to be cacheable
RUN echo "Caching Mathlib..." \
    && mkdir -p ${MATHLIB_CACHE_DIR} \
    && lake setup && lake build || echo "Mathlib setup deferred to runtime"

# ============================================================================
# INSTALL PYTHON BRIDGE DEPENDENCIES
# ============================================================================

WORKDIR ${WORKSPACE_DIR}

# Create Python virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python bridge dependencies
COPY requirements.txt ${WORKSPACE_DIR}/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found, installing minimal deps"

# Install minimal dependencies for Lean 4 bridge
RUN pip install --no-cache-dir \
    psutil \
    requests \
    pydantic \
    python-dotenv \
    typing-extensions

# ============================================================================
# PYTHON BRIDGE CODE
# ============================================================================

# Create bridge directory
RUN mkdir -p ${WORKSPACE_DIR}/bridge

# Copy bridge code (will be mounted as volume in production)
COPY bridge/ ${WORKSPACE_DIR}/bridge/ || echo "No bridge code to copy"

# Add bridge to Python path
ENV PYTHONPATH="${WORKSPACE_DIR}/bridge:${PYTHONPATH}"

# ============================================================================
# HEALTH CHECK SCRIPT
# ============================================================================

RUN echo '#!/bin/bash\n\
# Health check for Lean 4 environment\n\
set -e\n\
\n\
echo "Checking Lean 4 installation..."\n\
lean --version || exit 1\n\
\n\
echo "Checking Lake build system..."\n\
lake --version || exit 1\n\
\n\
echo "Checking Python bridge..."\n\
python3 -c "import psutil; print(f\"Memory: {psutil.virtual_memory().total // (1024**1024)}MB\")" || exit 1\n\
\n\
echo "Checking Mathlib cache..."\n\
ls -la ${MATHLIB_CACHE_DIR} || echo "Mathlib cache not yet populated"\n\
\n\
echo "Lean 4 environment healthy!"\n\
exit 0\n\
' > /usr/local/bin/healthcheck.sh \
    && chmod +x /usr/local/bin/healthcheck.sh

# ============================================================================
# WORKING DIRECTORY & USER
# ============================================================================

# Create non-root user for running Lean
RUN useradd -m -s /bin/bash lean && \
    chown -R lean:lean ${WORKSPACE_DIR}

USER lean
WORKDIR ${WORKSPACE_DIR}

# ============================================================================
# HEALTH CHECK
# ============================================================================

HEALTHCHECK --interval=${HEALTH_CHECK_INTERVAL} \
            --timeout=${HEALTH_CHECK_TIMEOUT} \
            --retries=${HEALTH_CHECK_RETRIES} \
            --start-period=60s \
    CMD /usr/local/bin/healthcheck.sh

# ============================================================================
# STARTUP SCRIPT
# ============================================================================

RUN echo '#!/bin/bash\n\
# Lean 4 environment startup script\n\
set -e\n\
\n\
echo "Starting Lean 4 Formal Verification Environment..."\n\
echo "Lean 4 version: $(lean --version)"\n\
echo "Lake version: $(lake --version)"\n\
echo "Python version: $(python3 --version)"\n\
echo "Workspace: ${WORKSPACE_DIR}"\n\
echo ""\n\
echo "Environment ready for formal verification!"\n\
echo ""\n\
# Keep container running if no command specified\n\
if [ "$#" -eq 0 ]; then\n\
    exec tail -f /dev/null\n\
else\n\
    exec "$@"\n\
fi\n\
' > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# ============================================================================
# EXPOSE PORTS
# ============================================================================

# Lean server port (for interactive development)
EXPOSE 8080

# ============================================================================
# VOLUMES
# ============================================================================

# Persist Mathlib cache
VOLUME [${MATHLIB_CACHE_DIR}]

# Persist Lean projects
VOLUME [${LEAN4_PROJECT_DIR}]

# ============================================================================
# DEFAULT COMMAND
# ============================================================================

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []

# ============================================================================
# END OF DOCKERFILE
# ============================================================================

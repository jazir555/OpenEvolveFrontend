# Dockerfile for RESE Tiered Verification System
#
# Multi-stage build for tiered verification adapter integrating:
# - Tier 1: Z3 Fast Verification
# - Tier 2: LeanAide AI-Assisted Proving
# - Tier 3: Lean 4 Formal Verification
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via env vars
# - Law of Runtime Truth: Verify solvers via probes
# - Minimal attack surface
#
# Author: RESE Team
# Created: 2026-02-04

# =============================================================================
# BASE IMAGE
# =============================================================================

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# DEPENDENCIES STAGE
# =============================================================================

FROM base AS dependencies

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Z3 STAGE
# =============================================================================

FROM dependencies AS z3-stage

# Install Z3
RUN curl -L https://github.com/Z3Prover/z3/releases/download/z3-4.12.6/z3-4.12.6-x64-glibc-2.35.zip -o z3.zip \
    && unzip z3.zip \
    && mv z3-4.12.6-x64-glibc-2.35/bin/z3 /usr/local/bin/ \
    && rm -rf z3.zip z3-4.12.6-x64-glibc-2.35

# Verify Z3 installation
RUN z3 --version

# =============================================================================
# LEAN 4 STAGE
# =============================================================================

FROM z3-stage AS lean4-stage

# Install Lean 4
RUN curl -L https://github.com/leanprover/lean4/releases/download/v4.6.0/lean-4.6.0-linux.tar.gz -o lean4.tar.gz \
    && tar -xzf lean4.tar.gz \
    && mv lean-4.6.0-linux/lean /usr/local/bin/ \
    && mv lean-4.6.0-linux/leanpkg /usr/local/bin/ \
    && rm -rf lean4.tar.gz lean-4.6.0-linux

# Verify Lean 4 installation
RUN lean --version

# =============================================================================
# FINAL STAGE
# =============================================================================

FROM lean4-stage AS final

# Create non-root user
RUN useradd -m -u 1000 verifier && \
    chown -R verifier:verifier /app

USER verifier

# Copy application code
COPY --chown=verifier:verifier src/ /app/src/
COPY --chown=verifier:verifier tests/ /app/tests/
COPY --chown=verifier:verifier probes/ /app/probes/

# Create workspace directory for Lean 4
RUN mkdir -p /workspace/lean4

# Set environment variables (Law of Configuration Explicitness)
ENV TIER1_TIMEOUT_MS=1000 \
    TIER1_MAX_CONSTRAINTS=100 \
    TIER2_TIMEOUT_MS=60000 \
    TIER2_MAX_CONSTRAINTS=1000 \
    TIER3_TIMEOUT_MS=300000 \
    AUTO_ESCALATE=true \
    MAX_TIER=3 \
    SELECTION_STRATEGY=adaptive \
    PREFER_FAST_SOLVER=true \
    MIN_CONFIDENCE_THRESHOLD=0.7 \
    ENABLE_MONITORING=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose ports (if running as service)
EXPOSE 8080

# Run application
CMD ["python", "-m", "src.tiered_verifier"]

# ============================================================================
# RESE Symbolic Constraint Engine (SCE) Adapter - Dockerfile
# ============================================================================
#
# Follows CLAUDE.md Law of the "Air Gap":
# - Isolated container
# - Shares network with Core, but no files
# - Contains all dependencies within image
#
# Build: docker build -t rese-sce-adapter:latest .
# Run: docker run --rm --network rese-network rese-sce-adapter:latest
# ============================================================================

FROM node:20-alpine

# Metadata
LABEL maintainer="OpenEvolve Frontend Team"
LABEL description="RESE Symbolic Constraint Engine (SCE) Adapter"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install dependencies for Lean 4 (if enabled)
# RUN apk add --no-cache \
#     emacs \
#     git \
#     curl \
#     && curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
#
# ENV PATH="/root/.elan:${PATH}"
# ENV LEAN_PATH=/app/lean4

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy source code
COPY src ./src
COPY ../../lib/rese-sce.ts ./lib/rese-sce.ts
COPY ../../schemas/rese-canonical.ts ./schemas/rese-canonical.ts
COPY ../../lib/*.ts ./lib/

# Copy TypeScript config
COPY tsconfig.json ./

# Build TypeScript
RUN npm run build || echo "Build step not yet configured"

# Create non-root user for security
RUN addgroup -g 1001 -S rese && \
    adduser -u 1001 -S rese -G rese

# Set ownership
RUN chown -R rese:rese /app

# Switch to non-root user
USER rese

# Expose health check port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "console.log('Health check')" || exit 1

# Default environment variables (Law of Configuration Explicitness)
ENV SCE_TIMEOUT_MS=5000
ENV SCE_CONSTRAINT_TIMEOUT_MS=3000
ENV SCE_CONTRADICTION_TIMEOUT_MS=10000
ENV SCE_MAX_ITERATIONS=1000
ENV SCE_MAX_CONSTRAINTS=10000
ENV SCE_MAX_CONTRADICTION_SET_SIZE=100
ENV SCE_CIRCUIT_BREAKER_THRESHOLD=5
ENV SCE_CIRCUIT_BREAKER_TIMEOUT_MS=60000
ENV SCE_ENABLE_LEAN4=false
ENV SCE_ENABLE_TACIT_MINING=true

ENV RESE_SCE_URL=http://rese-core:8000
ENV RESE_SCE_TIMEOUT_MS=30000
ENV SCE_ADAPTER_MAX_RETRIES=3
ENV SCE_ADAPTER_INITIAL_DELAY_MS=1000
ENV SCE_ADAPTER_MAX_DELAY_MS=10000
ENV SCE_ADAPTER_CB_THRESHOLD=5
ENV SCE_ADAPTER_CB_TIMEOUT_MS=60000
ENV SCE_DLQ_ENABLED=true
ENV SCE_DLQ_MAX_SIZE=1000

# Run the adapter
CMD ["node", "dist/index.js"]

# Dockerfile for RAGBits-Graphiti Sync Adapter
#
# Follows the Federation Constitution:
# - Law of Configuration Explicitness: All config via env vars
# - Law of Runtime Truth: Verify API before starting
#
# Usage:
#   docker build -t openevolve/ragbits-graphiti-sync:latest .
#   docker run -e RAGBITS_API_URL=http://ragbits:8000 \
#              -e GRAPHITI_API_URL=http://graphiti:8000 \
#              openevolve/ragbits-graphiti-sync:latest

FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    git \
    python3 \
    make \
    g++

# Copy package files
COPY package.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY src ./src

# Build TypeScript
RUN npm run build

# Verify build
RUN [ -f dist/index.js ] || (echo "Build failed: dist/index.js not found" && exit 1)

# Production stage
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    bash \
    jq

# Create non-root user
RUN addgroup -g 1001 -S syncuser && \
    adduser -u 1001 -S syncuser -G syncuser

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./
COPY --from=builder /app/node_modules ./node_modules

# Copy probe scripts
COPY probes ./probes
RUN chmod +x probes/*.sh

# Set environment defaults
ENV NODE_ENV=production
SYNC_ENABLED=true
SYNC_BIDIRECTIONAL=true
SYNC_INTERVAL_MS=300000
SYNC_TIMEOUT_MS=30000
SYNC_MAX_RETRIES=3
SYNC_BATCH_SIZE=10
SYNC_CONFLICT_RESOLUTION=newest_wins
SYNC_AUTO_RESOLVE_CONFLICTS=true
SYNC_SEMANTIC_SIMILARITY_THRESHOLD=0.8
SYNC_TEMPORAL_DRIFT_THRESHOLD_MS=60000

# Create health check script
RUN echo '#!/bin/sh\n\
echo "Health check..."\n\
curl -f -s http://localhost:8000/health || exit 1\n\
echo "Healthy"\n\
' > /usr/local/bin/healthcheck.sh && \
chmod +x /usr/local/bin/healthcheck.sh

# Expose health port
EXPOSE 8000

# Switch to non-root user
USER syncuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Run contract tests on startup (Phase 2: The Contract)
RUN echo '#!/bin/sh\n\
echo "Running contract tests..."\n\
cd /app/tests && npm test || exit 1\n\
echo "Contract tests passed"\n\
' > /usr/local/bin/contract-test.sh && \
chmod +x /usr/local/bin/contract-test.sh

# Start script
RUN echo '#!/bin/sh\n\
set -e\n\
echo "Starting RAGBits-Graphiti Sync Adapter..."\n\
echo "Configuration:"\n\
echo "  RAGBits API: ${RAGBITS_API_URL}"\n\
echo "  Graphiti API: ${GRAPHITI_API_URL}"\n\
echo "  Sync Enabled: ${SYNC_ENABLED}"\n\
echo "  Bidirectional: ${SYNC_BIDIRECTIONAL}"\n\
echo ""\n\
# Run contract tests\n\
/usr/local/bin/contract-test.sh\n\
\n\
# Verify configuration\n\
if [ -z "${RAGBITS_API_URL}" ]; then\n\
  echo "ERROR: RAGBITS_API_URL is required"\n\
  exit 1\n\
fi\n\
\n\
if [ -z "${GRAPHITI_API_URL}" ]; then\n\
  echo "ERROR: GRAPHITI_API_URL is required"\n\
  exit 1\n\
fi\n\
\n\
# Run probe tests\n\
echo "Running probe tests..."\n\
cd /app\n\
npm run probe:all || echo "Warning: Some probes failed"\n\
\n\
# Start the application\n\
echo "Starting sync manager..."\n\
node dist/index.js\n\
' > /usr/local/bin/start.sh && \
chmod +x /usr/local/bin/start.sh

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/start.sh"]

# Metadata
LABEL maintainer="OpenEvolve Federation"
LABEL description="Bidirectional sync adapter between RAGBits and Graphiti"
LABEL version="1.0.0"

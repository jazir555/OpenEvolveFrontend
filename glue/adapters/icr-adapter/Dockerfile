# ICR Adapter Dockerfile
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via environment variables
# - Law of Runtime Truth: Verify ICR modes on startup
# - Isolation: Adapter runs in separate container from ICR core

FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies for building
RUN apk add --no-cache \
    python3 \
    py3-pip \
    git

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src ./src

# Build TypeScript
RUN npm run build

# -----------------------------------------------------------------------------
# Production stage
# -----------------------------------------------------------------------------
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    bash

# Create app user
RUN addgroup -g 1001 -S icr && \
    adduser -S -u 1001 -G icr icr

# Set working directory
WORKDIR /app

# Copy built application from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

# Copy probe scripts
COPY probes ./probes
RUN chmod +x ./probes/*.sh 2>/dev/null || true

# Create log directory
RUN mkdir -p /var/log/icr-adapter && \
    chown -R icr:icr /var/log/icr-adapter

# Switch to non-root user
USER icr

# Environment variables (REQUIRED - service will crash if missing)
# ICR_API_URL         - ICR API endpoint (required)
# ICR_API_KEY         - ICR API key (optional)
# ICR_TIMEOUT_MS      - Request timeout in ms (default: 30000)
# ICR_MAX_RETRIES     - Max retry attempts (default: 3)
# ICR_RETRY_DELAY_MS  - Initial retry delay in ms (default: 1000)
# LOG_LEVEL           - Logging level (default: INFO)
# LOG_FORMAT          - Log format: json or text (default: json)
# SERVICE_PORT        - Service port (default: 3002)
# TZ                  - Timezone (default: UTC)

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD node -e "
    const http = require('http');
    const options = {
      host: 'localhost',
      port: process.env.SERVICE_PORT || 3002,
      path: '/health',
      timeout: 2000
    };
    const req = http.request(options, (res) => {
      process.exit(res.statusCode === 200 ? 0 : 1);
    });
    req.on('error', () => process.exit(1));
    req.end();
    " || exit 1

# Expose health check port
EXPOSE 3002

# Default command (can be overridden)
CMD ["node", "dist/index.js"]

# Labels
LABEL maintainer="OpenEvolve Federation" \
      version="1.0.0" \
      description="ICR (Iterative Contextual Refinements) adapter" \
      com.openevolve.icr-adapter.version="1.0.0"

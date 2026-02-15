# LeanAide Adapter Dockerfile
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via environment variables
# - Law of Runtime Truth: Verify LeanAide connection on startup
# - Isolation: Adapter runs in separate container from LeanAide core

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
RUN npm run build || echo "No build script, using ts-node"

# -----------------------------------------------------------------------------
# Production stage
# -----------------------------------------------------------------------------
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    bash

# Create app user
RUN addgroup -g 1001 -S leanaide && \
    adduser -S -u 1001 -G leanaide leanaide

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
RUN mkdir -p /var/log/leanaide-adapter && \
    chown -R leanaide:leanaide /var/log/leanaide-adapter

# Switch to non-root user
USER leanaide

# Environment variables (REQUIRED - service will crash if missing)
# LEANAIDE_API_URL      - LeanAide API endpoint (required)
# LEANAIDE_TIMEOUT_MS   - Request timeout in ms (default: 30000)
# LEANAIDE_MAX_RETRIES  - Max retry attempts (default: 3)
# LOG_LEVEL             - Logging level (default: INFO)
# LOG_FORMAT            - Log format: json or text (default: json)
# SERVICE_PORT          - Service port (default: 3006)
# TZ                    - Timezone (default: UTC)

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD node -e "
    const http = require('http');
    const options = {
      host: 'localhost',
      port: process.env.SERVICE_PORT || 3006,
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
EXPOSE 3006

# Default command (can be overridden)
CMD ["node", "dist/index.js"]

# Labels
LABEL maintainer="OpenEvolve Federation" \
      version="1.0.0" \
      description="LeanAide formal verification assistant adapter" \
      com.openevolve.leanaide-adapter.version="1.0.0"

# LoongFlow Adapter Dockerfile
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via environment variables
# - Law of Runtime Truth: Verify LoongFlow sidecar connection on startup
# - Isolation: Adapter runs in separate container from LoongFlow core

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
RUN addgroup -g 1001 -S loongflow && \
    adduser -S -u 1001 -G loongflow loongflow

# Set working directory
WORKDIR /app

# Copy built application from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

# Copy test fixtures and contract tests
COPY --from=builder /app/tests ./tests

# Copy probe scripts
COPY probes ./probes
RUN chmod +x ./probes/*.sh 2>/dev/null || true

# Create log directory
RUN mkdir -p /var/log/loongflow-adapter && \
    chown -R loongflow:loongflow /var/log/loongflow-adapter

# Switch to non-root user
USER loongflow

# Environment variables (REQUIRED - service will crash if missing)
# LOONGFLOW_API_URL     - LoongFlow sidecar API endpoint (required)
# LOONGFLOW_TIMEOUT_MS  - Request timeout in ms (default: 30000)
# LOONGFLOW_MAX_RETRIES - Max retry attempts (default: 3)
# LOG_LEVEL             - Logging level (default: info)
# LOG_FORMAT            - Log format: json or text (default: json)
# SERVICE_PORT          - Service port (default: 8040)
# TZ                    - Timezone (default: UTC)

# Health check - validates both adapter health and API contracts
# This ensures the adapter only runs if contracts are satisfied
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD node -e "
    const http = require('http');
    const options = {
      host: 'localhost',
      port: process.env.SERVICE_PORT || 8040,
      path: '/health',
      timeout: 2000
    };
    const req = http.request(options, (res) => {
      if (res.statusCode === 200) {
        // Run quick contract validation
        const { execSync } = require('child_process');
        try {
          execSync('npm run test:contract:quick', { stdio: 'ignore', timeout: 5000 });
          process.exit(0);
        } catch (e) {
          console.error('Contract validation failed');
          process.exit(1);
        }
      } else {
        process.exit(1);
      }
    });
    req.on('error', () => process.exit(1));
    req.end();
    " || exit 1

# Expose health check port
EXPOSE 8040

# Default command (can be overridden)
CMD ["node", "dist/index.js"]

# Labels
LABEL maintainer="OpenEvolve Federation" \
      version="1.0.0" \
      description="LoongFlow PES Agent adapter" \
      com.loongflow.loongflow-adapter.version="1.0.0"

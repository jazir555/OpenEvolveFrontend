# Graphiti Adapter Dockerfile
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via environment variables
# - Law of Runtime Truth: Verify Graphiti connection on startup
# - Isolation: Adapter runs in separate container from Graphiti core

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
    bash \
    cypher-shell \
    python3 \
    py3-pip

# Create app user
RUN addgroup -g 1001 -S graphiti && \
    adduser -S -u 1001 -G graphiti graphiti

# Set working directory
WORKDIR /app

# Copy built application from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

# Copy probe scripts
COPY probes ./probes
RUN chmod +x ./probes/*.sh

# Create log directory
RUN mkdir -p /var/log/graphiti-adapter && \
    chown -R graphiti:graphiti /var/log/graphiti-adapter

# Switch to non-root user
USER graphiti

# Environment variables (required)
# NEO4J_URI         - Neo4j connection URI
# NEO4J_USER        - Neo4j username
# NEO4J_PASSWORD    - Neo4j password
# OPENAI_API_KEY    - OpenAI API key (optional)
# TIMEOUT_MS        - Request timeout (default: 30000)

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "
    const http = require('http');
    const options = {
      host: 'localhost',
      port: 3000,
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
EXPOSE 3000

# Default command (can be overridden)
CMD ["node", "dist/index.js"]

# Labels
LABEL maintainer="OpenEvolve Federation" \
      version="1.0.0" \
      description="Graphiti temporal knowledge graph adapter" \
      com.openevolve.graphiti-adapter.version="1.0.0"

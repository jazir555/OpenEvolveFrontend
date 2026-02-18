# OpenEvolve React Plugin - Dockerfile
#
# Following Federation Constitution principles:
# - Law of Configuration Explicitness: All config via environment variables
# - Law of UTC: All timestamps in UTC
# - Health checks for monitoring

###############################################################################
# Build Stage
###############################################################################
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Set environment to production
ENV NODE_ENV=production
ENV TZ=UTC

# Install build dependencies
RUN apk add --no-cache \
    git \
    python3 \
    make \
    g++

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./
COPY vite.config.ts ./

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy source code
COPY src ./src

# Build the plugin
RUN npm run build && \
    # Verify build output exists
    test -d dist

###############################################################################
# Production Stage
###############################################################################
FROM node:18-alpine

# Set labels
LABEL maintainer="OpenEvolve Federation"
LABEL description="OpenEvolve React Plugin for BubbleLab Integration"
LABEL version="1.0.0"

# Set timezone to UTC (Law of UTC)
ENV TZ=UTC
ENV NODE_ENV=production

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    tzdata && \
    # Verify timezone is UTC
    date +%Z | grep -q UTC

# Create non-root user
RUN addgroup -g 1001 -S openevolve && \
    adduser -u 1001 -S openevolve -G openevolve

# Set working directory
WORKDIR /app

# Copy built files from builder
COPY --from=builder --chown=openevolve:openevolve /app/dist ./dist
COPY --from=builder --chown=openevolve:openevolve /app/node_modules ./node_modules
COPY --from=builder --chown=openevolve:openevolve /app/package.json ./package.json

# Switch to non-root user
USER openevolve

# Expose port (if running as dev server)
EXPOSE 3000

# Health check (Law of Runtime Truth)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "require('./dist/index.js')" || exit 1

# Default command (serve plugin files)
CMD ["node", "-e", "require('./dist/index.js')"]

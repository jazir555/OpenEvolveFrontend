# Auth Sidecar Dockerfile
# OAuth2-Proxy with custom configuration templates
#
# Following CLAUDE.md Federation Constitution:
# - Law of Configuration Explicitness: All config via environment variables
# - Law of Runtime Truth: Probe validates OIDC connectivity
# - Law of Idempotency: Stateless, safe to restart

FROM quay.io/oauth2-proxy/oauth2-proxy:v7.5.1

# Metadata
LABEL maintainer="OpenEvolve Federation <federation@openevolve.org>"
LABEL description="OAuth2-Proxy sidecar for central identity federation"
LABEL version="1.0.0"

# Set working directory
WORKDIR /etc/oauth2-proxy

# Copy custom configuration
COPY oauth2-proxy.cfg /etc/oauth2-proxy/oauth2-proxy.cfg

# Copy custom templates (if any)
COPY templates/ /etc/oauth2-proxy/templates/

# Copy health check probe
COPY probes/health-check.sh /usr/local/bin/health-check.sh
RUN chmod +x /usr/local/bin/health-check.sh

# Create directory for runtime config
RUN mkdir -p /var/run/oauth2-proxy

# Create non-root user for running the proxy
RUN useradd -r -s /bin/false oauth2-proxy && \
    chown -R oauth2-proxy:oauth2-proxy /etc/oauth2-proxy /var/run/oauth2-proxy

# Switch to non-root user
USER oauth2-proxy

# Expose ports
# 4180: Main proxy port
# 4181: Metrics port
EXPOSE 4180 4181

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /usr/local/bin/health-check.sh || exit 1

# Run OAuth2-Proxy
# Using envsubst to substitute environment variables in config
CMD ["sh", "-c", "envsubst < /etc/oauth2-proxy/oauth2-proxy.cfg > /var/run/oauth2-proxy/oauth2-proxy.cfg && /bin/oauth2-proxy --config /var/run/oauth2-proxy/oauth2-proxy.cfg"]

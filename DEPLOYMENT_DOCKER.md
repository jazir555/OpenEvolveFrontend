# Docker Deployment Guide

Complete Docker deployment guide for OpenEvolve with Hephaestus integration.

**Table of Contents:**
- [Docker Compose Setup](#docker-compose-setup)
- [Building Docker Images](#building-docker-images)
- [Container Orchestration](#container-orchestration)
- [Environment Variables](#environment-variables)
- [Volume Management](#volume-management)
- [Network Configuration](#network-configuration)
- [Health Checks](#health-checks)
- [Logging and Monitoring](#logging-and-monitoring)
- [Scaling Strategies](#scaling-strategies)

---

## Docker Compose Setup

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/openevolve.git
cd openevolve

# Copy environment file
cp .env.example .env

# Edit .env with your configuration
nano .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Complete Docker Compose File

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    container_name: openevolve_postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-openevolve}
      POSTGRES_USER: ${POSTGRES_USER:-openevolve}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./schema/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    ports:
      - "5432:5432"
    networks:
      - openevolve_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-openevolve}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Qdrant Vector Store
  qdrant:
    image: qdrant/qdrant:v1.7.0
    container_name: openevolve_qdrant
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC API
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - openevolve_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: openevolve_redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - openevolve_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # OpenEvolve API
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: openevolve_api
    environment:
      - ENV=${ENV:-production}
      - DATABASE_URL=postgresql://${POSTGRES_USER:-openevolve}:${POSTGRES_PASSWORD:-changeme}@postgres:5432/${POSTGRES_DB:-openevolve}
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_URL=http://qdrant:6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./openevolve:/app/openevolve:ro
      - cache_data:/app/cache
      - logs_data:/app/logs
    ports:
      - "8000:8000"
    networks:
      - openevolve_network
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # BubbleLab Frontend
  bubblelab:
    build:
      context: ./BubbleLab
      dockerfile: Dockerfile.bubble-studio
    container_name: openevolve_bubblelab
    environment:
      - VITE_API_URL=http://api:8000
      - VITE_APP_NAME=${VITE_APP_NAME:-OpenEvolve}
    volumes:
      - ./BubbleLab/apps/bubble-studio/src:/app/src:ro
    ports:
      - "3000:3000"
    networks:
      - openevolve_network
    depends_on:
      - api
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Hephaestus (Optional)
  hephaestus:
    build:
      context: ./Hephaestus
      dockerfile: Dockerfile
    container_name: openevolve_hephaestus
    environment:
      - HEPHAESTUS_API_KEY=${HEPHAESTUS_API_KEY}
      - HEPHAESTUS_API_BASE=http://api:8000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - QDRANT_URL=http://qdrant:6333
    volumes:
      - hephaestus_workspace:/workspace
      - hephaestus_cache:/app/.cache
    networks:
      - openevolve_network
    depends_on:
      - api
      - qdrant
    restart: unless-stopped
    profiles:
      - hephaestus  # Only start with --profile hephaestus

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: openevolve_nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    ports:
      - "80:80"
      - "443:443"
    networks:
      - openevolve_network
    depends_on:
      - api
      - bubblelab
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: openevolve_prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - openevolve_network
    restart: unless-stopped
    profiles:
      - monitoring

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: openevolve_grafana
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-changeme}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "3001:3000"
    networks:
      - openevolve_network
    depends_on:
      - prometheus
    restart: unless-stopped
    profiles:
      - monitoring

networks:
  openevolve_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres_data:
    driver: local
  qdrant_data:
    driver: local
  redis_data:
    driver: local
  cache_data:
    driver: local
  logs_data:
    driver: local
  hephaestus_workspace:
    driver: local
  hephaestus_cache:
    driver: local
  nginx_logs:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
```

---

## Building Docker Images

### API Dockerfile

Create `Dockerfile`:

```dockerfile
# Multi-stage build for OpenEvolve API
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    tmux \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/llm_cache

# Create non-root user
RUN useradd -m -u 1000 openevolve && \
    chown -R openevolve:openevolve /app

USER openevolve

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "openevolve_api.py"]
```

### Frontend Dockerfile

Create `BubbleLab/Dockerfile.bubble-studio`:

```dockerfile
# Build stage
FROM node:18-alpine as builder

WORKDIR /build

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build application
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from builder
COPY --from=builder /build/apps/bubble-studio/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost:3000/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
```

### Hephaestus Dockerfile

Create `Hephaestus/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    tmux \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create workspace directory
RUN mkdir -p /workspace /app/.cache

# Create non-root user
RUN useradd -m -u 1000 hephaestus && \
    chown -R hephaestus:hephaestus /app /workspace

USER hephaestus

# Set environment variables
ENV PYTHONPATH=/app
ENV WORKSPACE_DIR=/workspace

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run Hephaestus
CMD ["python", "-m", "hephaestus.main"]
```

### Build Commands

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build api

# Build with no cache
docker-compose build --no-cache

# Build with build args
docker-compose build --build-arg VITE_API_URL=https://api.example.com bubblelab
```

---

## Container Orchestration

### Docker Compose Commands

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d postgres qdrant

# Start with profile
docker-compose --profile monitoring up -d
docker-compose --profile hephaestus up -d

# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v

# Restart services
docker-compose restart

# Restart specific service
docker-compose restart api

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api

# Follow logs with tail
docker-compose logs -f --tail=100 api

# Execute command in container
docker-compose exec api bash
docker-compose exec postgres psql -U openevolve -d openevolve

# Run one-off command
docker-compose run api python scripts/migrate.py

# Show running processes
docker-compose top

# Show resource usage
docker stats
```

### Docker Swarm Deployment

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml openevolve

# List services
docker service ls

# Scale services
docker service scale openevolve_api=5

# Update service
docker service update openevolve_api --image openevolve/api:v2.0

# Rollback update
docker service rollback openevolve_api

# Remove stack
docker stack rm openevolve
```

### Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openevolve-api
  labels:
    app: openevolve-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openevolve-api
  template:
    metadata:
      labels:
        app: openevolve-api
    spec:
      containers:
      - name: api
        image: openevolve/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: openevolve-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openevolve-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: openevolve-api-service
spec:
  selector:
    app: openevolve-api
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: LoadBalancer
```

Deploy to Kubernetes:
```bash
# Create namespace
kubectl create namespace openevolve

# Create secrets
kubectl create secret generic openevolve-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=openai-api-key='sk-...' \
  -n openevolve

# Apply manifests
kubectl apply -f k8s/ -n openevolve

# Check deployment
kubectl get pods -n openevolve
kubectl get services -n openevolve

# View logs
kubectl logs -f deployment/openevolve-api -n openevolve

# Scale deployment
kubectl scale deployment openevolve-api --replicas=5 -n openevolve
```

---

## Environment Variables

### Docker-Specific Variables

Add to your `.env` file:

```bash
# Docker Configuration
DOCKER_REGISTRY=docker.io
DOCKER_IMAGE_PREFIX=openevolve
DOCKER_TAG=latest

# Resource Limits
API_MEMORY_LIMIT=1g
API_MEMORY_RESERVATION=512m
API_CPU_LIMIT=1
API_CPU_RESERVATION=0.5

POSTGRES_MEMORY_LIMIT=2g
POSTGRES_MEMORY_RESERVATION=1g

QDRANT_MEMORY_LIMIT=2g
QDRANT_MEMORY_RESERVATION=1g

# Network Configuration
NETWORK_SUBNET=172.20.0.0/16

# Volume Configuration
VOLUME_DRIVER=local
VOLUME_BACKUP_PATH=/backup/docker-volumes
```

### Secret Management

**Using Docker Secrets (Swarm):**

Create `docker-compose.swarm.yml`:
```yaml
version: '3.8'

services:
  api:
    image: openevolve/api:latest
    secrets:
      - openai_api_key
      - anthropic_api_key
      - database_url
      - secret_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_api_key
      - DATABASE_URL_FILE=/run/secrets/database_url
      - SECRET_KEY_FILE=/run/secrets/secret_key

secrets:
  openai_api_key:
    external: true
  anthropic_api_key:
    external: true
  database_url:
    external: true
  secret_key:
    external: true
```

Create secrets:
```bash
echo "sk-your-key" | docker secret create openai_api_key -
echo "sk-ant-your-key" | docker secret create anthropic_api_key -
echo "postgresql://..." | docker secret create database_url -
python -c 'import secrets; print(secrets.token_hex(32))' | docker secret create secret_key -
```

**Using Kubernetes Secrets:**

```bash
# Create secret from file
kubectl create secret generic openai-api-key --from-file=key.txt

# Create secret from literal
kubectl create secret generic database-url --from-literal=url='postgresql://...'

# Create TLS secret
kubectl create secret tls openevolve-tls --cert=path/to/cert.crt --key=path/to/cert.key

# Encode and create secret
echo -n 'sk-your-key' | base64
kubectl create secret generic openai-api-key --from-literal=key=<base64-encoded-key>
```

---

## Volume Management

### Backup Volumes

```bash
# Backup PostgreSQL volume
docker run --rm \
  --volumes-from openevolve_postgres \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/postgres_$(date +%Y%m%d).tar.gz /var/lib/postgresql/data

# Backup Qdrant volume
docker run --rm \
  --volumes-from openevolve_qdrant \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/qdrant_$(date +%Y%m%d).tar.gz /qdrant/storage

# Backup all volumes
docker run --rm \
  -v openevolve_postgres_data:/data/postgres \
  -v openevolve_qdrant_data:/data/qdrant \
  -v openevolve_redis_data:/data/redis \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/all_volumes_$(date +%Y%m%d).tar.gz /data
```

### Restore Volumes

```bash
# Restore PostgreSQL
docker run --rm \
  --volumes-from openevolve_postgres \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/postgres_20240101.tar.gz -C /

# Restore Qdrant
docker run --rm \
  --volumes-from openevolve_qdrant \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/qdrant_20240101.tar.gz -C /
```

### Volume Cleanup

```bash
# List unused volumes
docker volume ls -f dangling=true

# Remove unused volumes
docker volume prune

# Remove specific volume
docker volume rm openevolve_postgres_data

# Backup and remove all
docker volume ls -q | xargs -I {} sh -c 'docker run --rm -v {}:/data -v $(pwd)/backup:/backup alpine tar czf /backup/{}.tar.gz /data'
docker volume prune -f
```

---

## Network Configuration

### Custom Network

```yaml
networks:
  openevolve_network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
    driver_opts:
      com.docker.network.bridge.name: br-openevolve
      com.docker.network.bridge.enable_icc: "true"
      com.docker.network.bridge.enable_ip_masquerade: "true"
```

### External Network

```yaml
services:
  api:
    networks:
      - openevolve_internal
      - external_network

networks:
  openevolve_internal:
    internal: true
  external_network:
    external: true
    name: existing_network
```

### Network Security

```yaml
# Restrict inter-container communication
services:
  postgres:
    networks:
      - backend_only
    # No direct internet access

  api:
    networks:
      - backend_only
      - frontend

  nginx:
    networks:
      - frontend

networks:
  backend_only:
    internal: true
  frontend:
    driver: bridge
```

---

## Health Checks

### Custom Health Check Scripts

Create `scripts/healthcheck.sh`:

```bash
#!/bin/bash

# Check API health
curl -f http://localhost:8000/health || exit 1

# Check database connection
python -c "from sqlalchemy import create_engine; engine = create_engine('${DATABASE_URL}'); conn = engine.connect(); conn.close()" || exit 1

# Check Qdrant connection
curl -f ${QDRANT_URL}/ || exit 1

echo "All checks passed"
```

Make executable:
```bash
chmod +x scripts/healthcheck.sh
```

Use in Dockerfile:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD /app/scripts/healthcheck.sh
```

---

## Logging and Monitoring

### Centralized Logging

```yaml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service,environment"
```

### ELK Stack Integration

```yaml
services:
  elasticsearch:
    image: elasticsearch:8.0.0
    environment:
      - discovery.type=single-node
    volumes:
      - es_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.0.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    ports:
      - "5000:5000"
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.0.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

### Prometheus Metrics

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'openevolve-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
```

---

## Scaling Strategies

### Horizontal Scaling

```bash
# Scale API to 5 instances
docker-compose up -d --scale api=5

# Use HAProxy/Nginx for load balancing
# Add to docker-compose.yml:
haproxy:
  image: haproxy:alpine
  volumes:
    - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
  ports:
    - "80:80"
  depends_on:
    - api
```

### Vertical Scaling

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: openevolve-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: openevolve-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

**Next Steps:**
- Configure production settings in [PRODUCTION_CONFIG.md](PRODUCTION_CONFIG.md)
- Set up monitoring with [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md)
- Troubleshoot issues with [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Version:** 1.0.0
**Last Updated:** 2026-01-11

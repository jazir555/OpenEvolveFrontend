# Deployment Runbook

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Methods](#deployment-methods)
- [Environment Setup](#environment-setup)
- [Deployment Procedures](#deployment-procedures)
- [Rollback Procedures](#rollback-procedures)
- [Post-Deployment Verification](#post-deployment-verification)
- [Troubleshooting](#troubleshooting)

---

## Overview

This runbook covers the deployment of BubbleLab in various environments, including development, staging, and production. It provides step-by-step procedures for deploying using Docker, Kubernetes, and traditional hosting methods.

### Deployment Architecture

```mermaid
graph TB
    subgraph "Deployment Pipeline"
        Dev[Development]
        Staging[Staging]
        Prod[Production]
    end

    subgraph "Deployment Methods"
        Docker[Docker Compose]
        K8s[Kubernetes]
        Traditional[Traditional Hosting]
    end

    Dev --> Docker
    Dev --> Traditional
    Staging --> Docker
    Staging --> K8s
    Prod --> K8s
    Prod --> Docker

    style Dev fill:#e1f5ff
    style Staging fill:#fff4e1
    style Prod fill:#ffe1e1
```

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 20 GB

**Recommended Requirements:**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 50+ GB SSD

### Software Requirements

**For Docker Deployment:**
- Docker Engine 20.10+
- Docker Compose 2.0+

**For Kubernetes Deployment:**
- kubectl 1.25+
- Helm 3.0+ (optional)
- Kubernetes cluster 1.25+

**For Traditional Hosting:**
- Node.js 18+
- Bun 1.0+
- pnpm 8+
- PostgreSQL 14+
- Redis 6+

### Required Configuration

Before deploying, ensure you have:

1. **Domain Name** (for production)
2. **SSL Certificates** (Let's Encrypt or custom)
3. **API Keys** for AI services
4. **Database Credentials**
5. **Authentication Provider** (Clerk keys)
6. **Email Service** (Resend or similar)
7. **Object Storage** (Cloudflare R2 or S3)

---

## Deployment Methods

### 1. Docker Compose Deployment

#### Quick Start

```bash
# Clone repository
git clone https://github.com/bubblelabai/BubbleLab.git
cd BubbleLab

# Copy environment template
cp .env.template .env

# Edit environment variables
nano .env

# Build and start services
docker-compose up -d

# Check service status
docker-compose ps
```

#### Environment Configuration

Create a `.env` file in the project root:

```bash
# Application
BUBBLE_ENV=production
NODE_ENV=production

# Frontend
VITE_API_URL=https://api.yourdomain.com
VITE_CLERK_PUBLISHABLE_KEY=pk_live_...

# Backend
CLERK_SECRET_KEY=sk_live_...
DATABASE_URL=postgresql://user:password@postgres:5432/bubblelab
REDIS_URL=redis://redis:6379

# AI Services
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_KEY=your_openai_key

# Encryption
CREDENTIAL_ENCRYPTION_KEY=your_32_char_encryption_key

# Storage
CLOUDFLARE_R2_ACCESS_KEY=your_r2_access_key
CLOUDFLARE_R2_SECRET_KEY=your_r2_secret_key
CLOUDFLARE_R2_ACCOUNT_ID=your_account_id
CLOUDFLARE_R2_BUCKET_NAME=bubblelab-uploads

# Email
RESEND_API_KEY=your_resend_api_key

# Monitoring
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
```

#### Docker Compose Services

```yaml
version: '3.8'

services:
  # Frontend
  bubble-studio:
    build:
      context: .
      dockerfile: Dockerfile.studio
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=${VITE_API_URL}
      - VITE_CLERK_PUBLISHABLE_KEY=${VITE_CLERK_PUBLISHABLE_KEY}
    depends_on:
      - bubblelab-api
    restart: unless-stopped

  # Backend API
  bubblelab-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "3001:3001"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - CLERK_SECRET_KEY=${CLERK_SECRET_KEY}
      - CREDENTIAL_ENCRYPTION_KEY=${CREDENTIAL_ENCRYPTION_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  # Database
  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=bubblelab
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Reverse Proxy
  traefik:
    image: traefik:v2.10
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Deployment Steps

```bash
# 1. Pull latest code
git pull origin main

# 2. Build images
docker-compose build

# 3. Run database migrations
docker-compose exec bubblelab-api bun run migrate

# 4. Start services
docker-compose up -d

# 5. Check logs
docker-compose logs -f

# 6. Verify health
curl http://localhost:3001/health
```

---

### 2. Kubernetes Deployment

#### Prerequisites

```bash
# Verify kubectl connectivity
kubectl cluster-info
kubectl get nodes

# Create namespace
kubectl create namespace bubblelab

# Create secrets
kubectl create secret generic bubblelab-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=clerk-secret-key='sk_live_...' \
  --namespace=bubblelab
```

#### Deployment Manifests

**ConfigMap:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bubblelab-config
  namespace: bubblelab
data:
  BUBBLE_ENV: "production"
  NODE_ENV: "production"
  DATABASE_URL: "postgresql://postgres:5432/bubblelab"
  REDIS_URL: "redis://redis:6379"
```

**API Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bubblelab-api
  namespace: bubblelab
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bubblelab-api
  template:
    metadata:
      labels:
        app: bubblelab-api
    spec:
      containers:
      - name: bubblelab-api
        image: bubblelab/api:latest
        ports:
        - containerPort: 3001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: bubblelab-secrets
              key: database-url
        - name: CLERK_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: bubblelab-secrets
              key: clerk-secret-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: bubblelab-api
  namespace: bubblelab
spec:
  selector:
    app: bubblelab-api
  ports:
  - port: 3001
    targetPort: 3001
  type: ClusterIP
```

**Studio Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bubble-studio
  namespace: bubblelab
spec:
  replicas: 2
  selector:
    matchLabels:
      app: bubble-studio
  template:
    metadata:
      labels:
        app: bubble-studio
    spec:
      containers:
      - name: bubble-studio
        image: bubblelab/studio:latest
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "250m"
---
apiVersion: v1
kind: Service
metadata:
  name: bubble-studio
  namespace: bubblelab
spec:
  selector:
    app: bubble-studio
  ports:
  - port: 3000
    targetPort: 3000
  type: ClusterIP
```

**Ingress:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bubblelab-ingress
  namespace: bubblelab
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    traefik.ingress.kubernetes.io/router.middlewares: bubblelab-redirect-https@kubernetes
spec:
  tls:
  - hosts:
    - app.bubblelab.ai
    - api.bubblelab.ai
    secretName: bubblelab-tls
  rules:
  - host: app.bubblelab.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bubble-studio
            port:
              number: 3000
  - host: api.bubblelab.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bubblelab-api
            port:
              number: 3001
```

#### Deployment Steps

```bash
# 1. Apply namespace
kubectl apply -f k8s/namespace.yaml

# 2. Apply secrets
kubectl apply -f k8s/secrets.yaml

# 3. Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# 4. Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# 5. Deploy Redis
kubectl apply -f k8s/redis.yaml

# 6. Deploy API
kubectl apply -f k8s/api-deployment.yaml

# 7. Deploy Studio
kubectl apply -f k8s/studio-deployment.yaml

# 8. Apply Ingress
kubectl apply -f k8s/ingress.yaml

# 9. Verify deployment
kubectl get pods -n bubblelab
kubectl get services -n bubblelab
kubectl get ingress -n bubblelab

# 10. Check logs
kubectl logs -f deployment/bubblelab-api -n bubblelab
```

---

### 3. Traditional Hosting Deployment

#### Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Bun
curl -fsSL https://bun.sh/install | bash

# Install pnpm
npm install -g pnpm

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Install Redis
sudo apt install -y redis-server

# Install Nginx
sudo apt install -y nginx

# Install PM2
npm install -g pm2
```

#### Application Setup

```bash
# Clone repository
git clone https://github.com/bubblelabai/BubbleLab.git
cd BubbleLab

# Install dependencies
pnpm install

# Build packages
pnpm build:core
pnpm build

# Setup environment
cp .env.template .env
nano .env

# Run migrations
cd apps/bubblelab-api
bun run migrate

# Start API with PM2
pm2 start bun --name "bubblelab-api" -- run src/index.ts

# Build and start frontend
cd ../bubble-studio
pnpm build
pm2 start npm --name "bubble-studio" -- run preview

# Save PM2 configuration
pm2 save
pm2 startup
```

#### Nginx Configuration

```nginx
# /etc/nginx/sites-available/bubblelab

# API Server
server {
    listen 80;
    server_name api.bubblelab.ai;

    location / {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}

# Frontend
server {
    listen 80;
    server_name app.bubblelab.ai;

    root /var/www/bubble-studio;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Environment Setup

### Development Environment

```bash
# Quick start
pnpm install
pnpm run dev

# Services run on:
# - Frontend: http://localhost:3000
# - Backend: http://localhost:3001
```

### Staging Environment

```bash
# Deploy to staging
git checkout develop
git pull origin develop
docker-compose -f docker-compose.staging.yml up -d

# Or with Kubernetes
kubectl apply -f k8s/staging/
```

### Production Environment

```bash
# Deploy to production
git checkout main
git pull origin main

# Tag version
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Or with Kubernetes
kubectl apply -f k8s/production/
```

---

## Deployment Procedures

### Standard Deployment Flow

```mermaid
graph LR
    A[Code Commit] --> B[Run Tests]
    B --> C[Build Docker Images]
    C --> D[Push to Registry]
    D --> E[Deploy to Staging]
    E --> F[Run E2E Tests]
    F --> G[Manual QA]
    G --> H[Deploy to Production]
    H --> I[Monitor Metrics]
    I --> J[Success?]
    J -->|Yes| K[Complete]
    J -->|No| L[Rollback]
```

### Blue-Green Deployment

```bash
# 1. Deploy to green environment
kubectl apply -f k8s/production/ -l environment=green

# 2. Verify green environment
kubectl get pods -l environment=green
curl https://green.bubblelab.ai/health

# 3. Switch traffic to green
kubectl patch ingress bubblelab-ingress -p '{"spec":{"rules":[{"host":"api.bubblelab.ai","http":{"paths":[{"backend":{"service":{"name":"bubblelab-api-green"}}}]}}]}'

# 4. Monitor green environment
# Check metrics, logs, error rates

# 5. If successful, remove blue environment
kubectl delete -f k8s/production/ -l environment=blue

# 6. If failed, rollback to blue
kubectl patch ingress bubblelab-ingress -p '{"spec":{"rules":[{"host":"api.bubblelab.ai","http":{"paths":[{"backend":{"service":{"name":"bubblelab-api-blue"}}}]}}]}'
```

### Canary Deployment

```bash
# 1. Deploy canary version (10% traffic)
kubectl apply -f k8s/canary/

# 2. Update service to send 10% to canary
kubectl patch svc bubblelab-api -p '{"spec":{"selector":{"version":"canary"}}}'

# 3. Monitor canary metrics
# - Error rates
# - Response times
# - Resource usage

# 4. Gradually increase traffic (10% -> 25% -> 50% -> 100%)

# 5. If successful, promote canary to production
kubectl apply -f k8s/production/

# 6. If failed, rollback
kubectl rollout undo deployment/bubblelab-api
```

---

## Rollback Procedures

### Docker Rollback

```bash
# 1. List previous versions
docker images | grep bubblelab

# 2. Stop current containers
docker-compose down

# 3. Update docker-compose.yml with previous image version
# image: bubblelab/api:v1.0.1  # rollback from v1.0.2

# 4. Start previous version
docker-compose up -d

# 5. Verify rollback
curl http://localhost:3001/health
docker-compose logs
```

### Kubernetes Rollback

```bash
# 1. Check rollout history
kubectl rollout history deployment/bubblelab-api -n bubblelab

# 2. Rollback to previous version
kubectl rollout undo deployment/bubblelab-api -n bubblelab

# 3. Rollback to specific revision
kubectl rollout undo deployment/bubblelab-api -n bubblelab --to-revision=3

# 4. Verify rollback
kubectl get pods -n bubblelab
kubectl logs -f deployment/bubblelab-api -n bubblelab

# 5. Check rollout status
kubectl rollout status deployment/bubblelab-api -n bubblelab
```

### Database Rollback

```bash
# 1. List migrations
bun run migrate:status

# 2. Rollback last migration
bun run migrate:rollback

# 3. Rollback specific migration
bun run migrate:rollback --to=20240101000000

# 4. Verify database schema
bun run migrate:status
```

---

## Post-Deployment Verification

### Health Checks

```bash
# API Health Check
curl https://api.bubblelab.ai/health

# Expected Response:
{
  "status": "healthy",
  "timestamp": "2026-01-18T10:00:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "ai": "operational"
  }
}

# Frontend Health Check
curl -I https://app.bubblelab.ai

# Expected Response:
HTTP/2 200
content-type: text/html
```

### Smoke Tests

```bash
# 1. Test authentication
curl -X POST https://api.bubblelab.ai/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password"}'

# 2. Test workflow creation
curl -X POST https://api.bubblelab.ai/api/bubble-flows \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Flow","bubbles":[]}'

# 3. Test workflow execution
curl -X POST https://api.bubblelab.ai/api/execute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"flowId":"...","payload":{}}'
```

### Monitoring Checks

```bash
# 1. Check pod status
kubectl get pods -n bubblelab

# 2. Check resource usage
kubectl top pods -n bubblelab
kubectl top nodes

# 3. Check logs
kubectl logs -f deployment/bubblelab-api -n bubblelab --tail=100

# 4. Check metrics
curl http://prometheus:9090/api/v1/query?query=up{job="bubblelab-api"}

# 5. Check traces
curl http://jaeger:16686/api/traces?service=bubblelab-api
```

### Performance Verification

```bash
# 1. Run load test
 artillery run load-test.yml

# 2. Check response times
curl -w "@curl-format.txt" -o /dev/null -s https://api.bubblelab.ai/health

# curl-format.txt:
# time_namelookup: %{time_namelookup}\n
# time_connect: %{time_connect}\n
# time_appconnect: %{time_appconnect}\n
# time_pretransfer: %{time_pretransfer}\n
# time_starttransfer: %{time_starttransfer}\n
# time_total: %{time_total}\n

# 3. Verify resource limits
kubectl describe pod bubblelab-api-xxxxx -n bubblelab
```

---

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

**Symptoms:**
- Pod stuck in `CrashLoopBackOff`
- Container exits immediately

**Diagnosis:**
```bash
# Check logs
kubectl logs -f deployment/bubblelab-api -n bubblelab

# Check events
kubectl describe pod bubblelab-api-xxxxx -n bubblelab

# Check resource usage
kubectl top pods -n bubblelab
```

**Solutions:**
- Check environment variables
- Verify database connectivity
- Increase resource limits
- Check for port conflicts

#### 2. Database Connection Issues

**Symptoms:**
- API returns 500 errors
- Logs show "ECONNREFUSED"

**Diagnosis:**
```bash
# Test database connection
psql -h postgres -U postgres -d bubblelab

# Check database pod
kubectl get pods -n bubblelab -l app=postgres
kubectl logs -f deployment/postgres -n bubblelab
```

**Solutions:**
- Verify DATABASE_URL
- Check database credentials
- Ensure database is ready
- Check network policies

#### 3. High Memory Usage

**Symptoms:**
- OOMKilled events
- Pod restarts

**Diagnosis:**
```bash
# Check memory usage
kubectl top pods -n bubblelab

# Check memory limits
kubectl describe pod bubblelab-api-xxxxx -n bubblelab

# Check memory profile
kubectl exec -it bubblelab-api-xxxxx -n bubblelab -- bun run profile
```

**Solutions:**
- Increase memory limits
- Optimize code
- Enable caching
- Add more replicas

#### 4. Slow Response Times

**Symptoms:**
- High latency
- Timeout errors

**Diagnosis:**
```bash
# Check response times
curl -w "@curl-format.txt" https://api.bubblelab.ai/health

# Check database query performance
kubectl exec -it postgres-0 -n bubblelab -- psql -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check Redis performance
kubectl exec -it redis-0 -n bubblelab -- redis-cli INFO stats
```

**Solutions:**
- Add database indexes
- Enable Redis caching
- Add more replicas
- Optimize slow queries

---

## Deployment Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Environment variables configured
- [ ] Database backups created
- [ ] Rollback plan documented
- [ ] Team notified of deployment
- [ ] Monitoring dashboards ready

### During Deployment

- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Check error rates
- [ ] Monitor resource usage
- [ ] Verify all services healthy

### Post-Deployment

- [ ] Run full test suite
- [ ] Check application logs
- [ ] Monitor metrics (15+ minutes)
- [ ] Verify key user flows
- [ ] Check alerting rules
- [ ] Update documentation
- [ ] Notify team of completion

---

## Related Documentation

- [troubleshooting.md](./troubleshooting.md) - Common issues and solutions
- [scaling.md](./scaling.md) - Scaling procedures
- [backup-recovery.md](./backup-recovery.md) - Backup and recovery
- [monitoring.md](./monitoring.md) - Monitoring and alerting
- [maintenance.md](./maintenance.md) - Routine maintenance procedures

---

*Last Updated: January 2026*

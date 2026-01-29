# 🚀 OpenEvolve Deployment Guide

Complete production deployment infrastructure for the OpenEvolve platform.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Deployment Procedures](#deployment-procedures)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Rollback Procedures](#rollback-procedures)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Nginx (Port 80/443)                   │
│                    (Reverse Proxy + SSL + Rate Limit)        │
└──────────────┬────────────────────────────┬─────────────────┘
               │                            │
               ▼                            ▼
    ┌──────────────────┐        ┌──────────────────┐
    │   BubbleLab      │        │   API Gateway    │
    │   Frontend       │        │   (FastAPI)      │
    │   (Port 3000)    │        │   (Port 8000)    │
    └──────────────────┘        └────────┬─────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            ┌───────────┐         ┌───────────┐         ┌───────────┐
            │ PostgreSQL│         │   Redis   │         │Knowledge  │
            │  (DB)     │         │  (Cache)  │         │  Engine   │
            └───────────┘         └───────────┘         └───────────┘
```

### Components

- **Nginx**: Reverse proxy, SSL termination, rate limiting, load balancing
- **BubbleLab Frontend**: React/TypeScript UI served from Vite dev server
- **API Gateway**: FastAPI backend with 87 REST endpoints + 5 WebSocket channels
- **PostgreSQL**: Primary data store
- **Redis**: Caching, session management, WebSocket pub/sub
- **Monitoring Stack**: Prometheus + Grafana + AlertManager

---

## 📦 Prerequisites

### Required Software

- Docker 20.10+
- Docker Compose 2.0+
- OpenSSL (for SSL certificates)
- Bash 4.0+
- curl, wget

### System Requirements

- **Minimum**: 4 CPU cores, 8GB RAM, 50GB storage
- **Recommended**: 8 CPU cores, 16GB RAM, 100GB storage
- **Production**: 16+ CPU cores, 32GB+ RAM, SSD storage

### Network Requirements

- Ports 80, 443 (HTTP/HTTPS)
- Port 8000 (API Gateway - internal)
- Port 3000 (Frontend - internal)
- Port 5432 (PostgreSQL - internal)
- Port 6379 (Redis - internal)
- Port 9090 (Prometheus - internal)
- Port 3001 (Grafana - internal)

---

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Clone repository
git clone https://github.com/openevolve/frontend.git
cd frontend

# Run setup script
bash deploy/scripts/setup.sh
```

This creates all necessary directories, generates SSL certificates, and creates environment file templates.

### 2. Configure Environment

```bash
# For staging
cp deploy/staging/.env.staging.example deploy/staging/.env.staging
nano deploy/staging/.env.staging

# For production
cp deploy/production/.env.production.example deploy/production/.env.production
nano deploy/production/.env.production
```

**Critical variables to configure:**

```bash
# Database
DB_PASSWORD=<generate secure password>
DATABASE_URL=<production only - external DB>

# Security
JWT_SECRET=<generate 32+ character secret>

# Frontend
VITE_API_BASE_URL=<API endpoint URL>
VITE_WS_BASE_URL=<WebSocket endpoint URL>

# Monitoring
GRAFANA_PASSWORD=<secure Grafana password>
```

### 3. Deploy to Staging

```bash
bash deploy/scripts/deploy-staging.sh
```

### 4. Verify Deployment

```bash
# Health checks
bash deploy/scripts/health-check.sh staging

# Smoke tests
bash deploy/scripts/smoke-tests.sh staging
```

---

## 🔧 Environment Setup

### Staging Environment

**Purpose**: Testing environment for pre-production validation

**Access**:
- Frontend: http://staging.openevolve.ai
- API: http://staging.openevolve.ai/api
- API Docs: http://staging.openevolve.ai/docs
- Grafana: http://staging.openevolve.ai:3001

**Configuration**: `deploy/staging/docker-compose.staging.yml`

**Features**:
- Single replica of each service
- Local PostgreSQL/Redis volumes
- Self-signed SSL certificates
- DEBUG/INFO logging
- Hot reloading enabled

### Production Environment

**Purpose**: Live production environment

**Access**:
- Frontend: https://openevolve.ai
- API: https://openevolve.ai/api
- API Docs: https://openevolve.ai/docs (restricted)
- Grafana: http://monitoring.openevolve.ai:3001

**Configuration**: `deploy/production/docker-compose.production.yml`

**Features**:
- 3 replicas of API Gateway
- External managed PostgreSQL (AWS RDS, etc.)
- External managed Redis (AWS ElastiCache, etc.)
- Let's Encrypt SSL certificates
- ERROR/WARN logging only
- Resource limits enforced
- Health checks required

---

## 📋 Deployment Procedures

### Pre-Deployment Checklist

Before deploying to any environment, complete the checklist at:

```bash
deploy/checklists/pre-deployment.md
```

Key checks:
- [ ] All tests passing locally
- [ ] Environment variables configured
- [ ] Database migrations prepared
- [ ] Backups enabled
- [ ] Monitoring configured
- [ ] Rollback plan ready

### Staging Deployment

```bash
# 1. Ensure you're on main branch
git checkout main
git pull origin main

# 2. Run pre-deployment checks
bash deploy/scripts/smoke-tests.sh local

# 3. Deploy to staging
bash deploy/scripts/deploy-staging.sh

# 4. Verify deployment
bash deploy/scripts/health-check.sh staging
bash deploy/scripts/smoke-tests.sh staging

# 5. View logs
docker-compose -f deploy/staging/docker-compose.staging.yml logs -f
```

### Production Deployment

```bash
# 1. Create a deployment branch
git checkout -b deploy-$(date +%Y%m%d)

# 2. Deploy to staging first
git checkout main
bash deploy/scripts/deploy-staging.sh
bash deploy/scripts/smoke-tests.sh staging

# 3. If staging passes, deploy to production
bash deploy/scripts/deploy-production.sh

# 4. Monitor deployment
watch -n 5 'bash deploy/scripts/health-check.sh production'

# 5. Run smoke tests
bash deploy/scripts/smoke-tests.sh production

# 6. Check post-deployment checklist
cat deploy/checklists/post-deployment.md
```

### Zero-Downtime Deployment

Production deployments use rolling updates:

```bash
# Deploy with zero downtime
docker-compose -f deploy/production/docker-compose.production.yml \
  up -d --no-deps --build api-gateway

# Watch health status
docker ps
```

The API Gateway has 3 replicas. Each replica is updated one at a time with health checks between updates.

---

## 📊 Monitoring

### Stack Components

- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Visualization (port 3001)
- **AlertManager**: Alert routing and notification (port 9093)

### Starting Monitoring Stack

```bash
docker-compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
```

### Accessing Dashboards

```
URL: http://localhost:3001 (or http://monitoring.openevolve.ai:3001)
Username: admin
Password: <configured in .env.production>
```

### Key Dashboards

1. **API Gateway Metrics**
   - Request rate, error rate, latency
   - WebSocket connections
   - CPU/memory usage

2. **Database Metrics**
   - Connection pool usage
   - Query performance
   - Replication lag

3. **Frontend Metrics**
   - Page load times
   - API call performance
   - Error rates

### Setting Up Alerts

Edit `deploy/monitoring/alertmanager.yml`:

```yaml
receivers:
  - name: 'email-notifications'
    email_configs:
      - to: 'ops@openevolve.ai'
        from: 'alertmanager@openevolve.ai'
        smarthost: 'smtp.gmail.com:587'
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker-compose -f deploy/[env]/docker-compose.[env].yml logs [service]

# Example
docker-compose -f deploy/production/docker-compose.production.yml logs api-gateway
```

#### 2. Database Connection Failed

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check connection
docker-compose -f deploy/production/docker-compose.production.yml \
  exec postgres psql -U openvolve -d openvolve_prod -c "SELECT 1;"

# Check logs
docker-compose -f deploy/production/docker-compose.production.yml logs postgres
```

#### 3. High Memory Usage

```bash
# Check container stats
docker stats

# Restart specific service
docker-compose -f deploy/production/docker-compose.production.yml restart api-gateway

# Scale down if needed
docker-compose -f deploy/production/docker-compose.production.yml \
  up -d --scale api-gateway=2
```

#### 4. SSL Certificate Issues

```bash
# Check certificate expiration
openssl x509 -in deploy/staging/ssl/cert.pem -noout -dates

# Regenerate self-signed cert
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deploy/staging/ssl/key.pem \
  -out deploy/staging/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=staging.openevolve.ai"
```

#### 5. WebSocket Connection Failed

```bash
# Check WebSocket endpoint
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  http://localhost:8000/ws/evolution

# Check Nginx config
docker-compose -f deploy/production/docker-compose.production.yml \
  exec nginx nginx -t

# Reload Nginx
docker-compose -f deploy/production/docker-compose.production.yml \
  exec nginx nginx -s reload
```

### Log Locations

```bash
# Container logs
docker-compose -f deploy/[env]/docker-compose.[env].yml logs -f

# Nginx access logs
docker-compose -f deploy/[env]/docker-compose.[env].yml \
  exec nginx tail -f /var/log/nginx/access.log

# Nginx error logs
docker-compose -f deploy/[env]/docker-compose.[env].yml \
  exec nginx tail -f /var/log/nginx/error.log

# API Gateway logs
docker-compose -f deploy/[env]/docker-compose.[env].yml \
  logs -f api-gateway
```

---

## 🔄 Rollback Procedures

### Automatic Rollback

If smoke tests fail during deployment, the script automatically rolls back:

```bash
# This happens automatically in deploy-production.sh
if ! bash deploy/scripts/smoke-tests.sh production; then
  bash deploy/scripts/rollback.sh
fi
```

### Manual Rollback

```bash
# 1. List available backups
ls -lt backups/

# 2. Run rollback script
bash deploy/scripts/rollback.sh

# 3. Select backup to restore
# (Enter backup number when prompted)

# 4. Verify rollback
bash deploy/scripts/health-check.sh production
```

### Database Rollback

```bash
# 1. Stop application
docker-compose -f deploy/production/docker-compose.production.yml stop api-gateway

# 2. Restore database
docker-compose -f deploy/production/docker-compose.production.yml \
  run --rm postgres psql < backups/production_20250106_120000/database.sql

# 3. Restart application
docker-compose -f deploy/production/docker-compose.production.yml start api-gateway
```

### Code Rollback

```bash
# 1. Revert to previous commit
git log --oneline -10
git revert HEAD

# 2. Push revert
git push origin main

# 3. Redeploy
bash deploy/scripts/deploy-production.sh
```

---

## 🧪 Testing

### Running Tests Locally

```bash
# Backend tests
cd api/gateway
pytest tests/ -v --cov=. --cov-report=html

# Frontend tests
cd OpenEvolve-Plugin
npm run test
```

### Running Smoke Tests

```bash
# Local
bash deploy/scripts/smoke-tests.sh local http://localhost:8000

# Staging
bash deploy/scripts/smoke-tests.sh staging

# Production
bash deploy/scripts/smoke-tests.sh production
```

### Load Testing

```bash
# Install k6
curl https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz -L | tar xvz

# Run load test
./k6 run tests/load/api-load-test.js
```

---

## 📝 Maintenance

### Regular Tasks

**Daily:**
- Check Grafana dashboards for anomalies
- Review error logs
- Verify backup completion

**Weekly:**
- Review and rotate logs
- Update SSL certificates (if using Let's Encrypt)
- Check disk space usage

**Monthly:**
- Database vacuum and analyze
- Review and update dependencies
- Security audit
- Capacity planning

### Backup Strategy

**Automated Backups:**

```bash
# Add to crontab
0 2 * * * /path/to/deploy/scripts/backup.sh production
```

**Manual Backup:**

```bash
# Database
docker-compose -f deploy/production/docker-compose.production.yml \
  exec postgres pg_dumpall -U openvolve > backup_$(date +%Y%m%d).sql

# Volumes
docker run --rm -v openevolve_prod_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/volume_backup_$(date +%Y%m%d).tar.gz /data
```

---

## 🔐 Security

### Best Practices

1. **Environment Variables**: Never commit `.env` files to git
2. **Secrets Management**: Use Docker Secrets or external vault (HashiCorp Vault, AWS Secrets Manager)
3. **SSL/TLS**: Always use HTTPS in production
4. **Firewall**: Restrict access to internal ports
5. **Updates**: Keep Docker images updated
6. **Scanning**: Regular vulnerability scans

### Security Checklist

- [ ] Change default passwords
- [ ] Enable SSL/TLS
- [ ] Configure firewall rules
- [ ] Set up fail2ban
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Implement rate limiting
- [ ] Configure CORS properly
- [ ] Use secrets manager
- [ ] Regular backups

---

## 📞 Support

For issues or questions:

- **Documentation**: See `README.md` and `ARCHITECTURE.md`
- **Issues**: https://github.com/openevolve/frontend/issues
- **Email**: ops@openevolve.ai

---

## 📄 License

Copyright © 2025 OpenEvolve. All rights reserved.

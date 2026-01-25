# BubbleLab Production Deployment Checklist

## Overview

This checklist ensures a safe, reliable production deployment process. Each item must be checked and documented before, during, and after deployment.

**Deployment Target**: BubbleLab Production System
**Required Approval**: Engineering Lead + DevOps Lead
**Deployment Window**: _______________
**Deployment Engineer**: _______________

---

## Table of Contents

1. [Pre-Deployment Checks](#pre-deployment-checks)
2. [Deployment Steps](#deployment-steps)
3. [Post-Deployment Verification](#post-deployment-verification)
4. [Rollback Procedures](#rollback-procedures)
5. [Health Check Endpoints](#health-check-endpoints)
6. [Graceful Shutdown Procedures](#graceful-shutdown-procedures)

---

## Pre-Deployment Checks

### 1. Code Review ✅

- [ ] **All code changes reviewed**
  - [ ] At least 2 approvals required
  - [ ] No outstanding review comments
  - [ ] Security review completed for sensitive changes

- [ ] **Tests passing**
  - [ ] Unit tests: 100% passing
  - [ ] Integration tests: 100% passing
  - [ ] Contract tests: 100% passing
  - [ ] E2E tests: 100% passing

- [ ] **Documentation updated**
  - [ ] API documentation updated
  - [ ] README updated
  - [ ] CHANGELOG updated
  - [ ] Migration guide created (if breaking changes)

### 2. Security Checks ✅

- [ ] **Security scan passed**
  - [ ] No critical vulnerabilities
  - [ ] No high vulnerabilities (or documented exceptions)
  - [ ] Dependencies up to date

- [ ] **Secrets management verified**
  - [ ] No secrets in code
  - [ ] No secrets in repository
  - [ ] Environment variables documented
  - [ ] Secrets rotated (if needed)

- [ ] **Security checklist reviewed**
  - [ ] File: `docs/SECURITY_CHECKLIST.md`
  - [ ] All critical items addressed
  - [ ] Authentication tested
  - [ ] Rate limiting tested
  - [ ] Input validation tested

### 3. Configuration Validation ✅

- [ ] **Environment variables validated**
  ```bash
  node config/validate-config.js --env production --strict
  ```
  - [ ] Exit code 0 (success)
  - [ ] No warnings

- [ ] **Configuration files reviewed**
  - [ ] `config/environments/production.yaml`
  - [ ] `config/service-discovery.yaml`
  - [ ] `config/workflow-registry.yaml`
  - [ ] Docker Compose files

- [ ] **Service discovery verified**
  - [ ] All service URLs reachable
  - [ ] All API keys present
  - [ ] All credentials present

### 4. Infrastructure Checks ✅

- [ ] **Infrastructure ready**
  - [ ] Servers provisioned
  - [ ] Docker installed
  - [ ] Docker Compose installed
  - [ ] Network configured

- [ ] **Resources available**
  - [ ] CPU: > 50% available
  - [ ] Memory: > 50% available
  - [ ] Disk: > 30% available
  - [ ] Network: Bandwidth sufficient

- [ ] **Dependencies ready**
  - [ ] PostgreSQL: Running and healthy
  - [ ] Qdrant: Running and healthy
  - [ ] Elasticsearch: Running and healthy
  - [ ] Redis: Running and healthy

### 5. Database Checks ✅

- [ ] **Migrations prepared**
  - [ ] Migration files created
  - [ ] Migrations tested in staging
  - [ ] Rollback migrations prepared
  - [ ] Migration estimated duration: ______ minutes

- [ ] **Backup verified**
  - [ ] Latest backup exists (< 24 hours)
  - [ ] Backup integrity verified
  - [ ] Restore test passed (last 30 days)

- [ ] **Database capacity checked**
  - [ ] Sufficient storage for growth
  - [ ] Connection pool size configured
  - [ ] Query performance tested

### 6. Monitoring Setup ✅

- [ ] **Monitoring configured**
  - [ ] Prometheus targets configured
  - [ ] Grafana dashboards created
  - [ ] Alert rules configured
  - [ ] Alertmanager routing tested

- [ ] **Log aggregation ready**
  - [ ] Loki configured
  - [ ] Promtail configured
  - [ ] Log streaming verified

- [ ] **Alerts configured**
  - [ ] PagerDuty integration tested
  - [ ] Slack integration tested
  - [ ] Email integration tested
  - [ ] On-call team notified

### 7. Performance Checks ✅

- [ ] **Load testing completed**
  - [ ] Normal load test passed (100 req/s)
  - [ ] Peak load test passed (500 req/s)
  - [ ] Stress test completed (1000 req/s)
  - [ ] Baselines documented

- [ ] **Performance benchmarks met**
  - [ ] Response time p95 < 500ms (APIs)
  - [ ] Response time p95 < 5000ms (workflows)
  - [ ] Error rate < 1%
  - [ ] Throughput > 1000 req/min

### 8. Documentation Checks ✅

- [ ] **Runbooks reviewed**
  - [ ] Incident runbooks available
  - [ ] Deployment runbook available
  - [ ] Rollback runbook available
  - [ ] On-call team trained

- [ ] **Communication prepared**
  - [ ] Stakeholders notified
  - [ ] Maintenance window announced
  - [ ] Status page updated (if applicable)
  - [ ] Support team briefed

### 9. Backup & Recovery ✅

- [ ] **Pre-deployment backup created**
  ```bash
  ./scripts/backup-all.sh
  ```
  - [ ] PostgreSQL backed up
  - [ ] Qdrant snapshots created
  - [ ] Elasticsearch snapshots created
  - [ ] Redis RDB backed up
  - [ ] Configuration backed up

- [ ] **Restore test passed** (last 7 days)
  - [ ] PostgreSQL restore tested
  - [ ] Qdrant restore tested
  - [ ] Elasticsearch restore tested
  - [ ] Redis restore tested

### 10. Rollback Plan ✅

- [ ] **Rollback procedure documented**
  - [ ] Steps documented in this checklist
  - [ ] Rollback command tested in staging
  - [ ] Rollback time estimated: ______ minutes

- [ ] **Rollback triggers defined**
  - [ ] Error rate > 5% for 5 minutes
  - [ ] Response time p95 > 10s for 5 minutes
  - [ ] Critical services down
  - [ ] Data corruption detected

- [ ] **Rollback verification**
  - [ ] Previous version tagged
  - [ ] Previous database schema known
  - [ ] Previous configuration saved

---

## Deployment Steps

### Phase 1: Preparation (T-30 minutes)

1. **Notify team** (T-30 minutes)
   ```bash
   # Post to Slack
   "@channel Deployment starting in 30 minutes. System may be briefly unavailable."
   ```

2. **Create pre-deployment backup** (T-25 minutes)
   ```bash
   ./scripts/backup-all.sh

   # Verify backup
   ./scripts/verify-backup.sh
   ```

3. **Stop non-critical services** (T-20 minutes)
   ```bash
   docker-compose stop some-non-critical-service
   ```

4. **Enable maintenance mode** (T-15 minutes)
   ```bash
   # Update load balancer or API gateway
   curl -X POST http://load-balancer/maintenance -d '{"enabled": true}'
   ```

### Phase 2: Database Migration (T-10 minutes)

1. **Run database migrations**
   ```bash
   # Dry run first
   npm run migrate:dry-run

   # Actual migration
   npm run migrate:up

   # Verify migration
   npm run migrate:status
   ```

2. **Verify data integrity**
   ```bash
   npm run verify-data
   ```

3. **Update database schema documentation**
   ```bash
   npm run generate-schema-docs
   ```

### Phase 3: Application Deployment (T-5 minutes)

1. **Build new version**
   ```bash
   # Build Docker images
   docker-compose build

   # Tag images
   docker tag bubblelab-api:latest bubblelab-api:${VERSION}
   ```

2. **Deploy new version**
   ```bash
   # Stop old version
   docker-compose stop bubblelab-api

   # Start new version
   docker-compose up -d bubblelab-api

   # Wait for startup
   sleep 30
   ```

3. **Run smoke tests**
   ```bash
   npm run smoke-test
   ```

### Phase 4: Verification (T+0 minutes)

1. **Disable maintenance mode**
   ```bash
   curl -X POST http://load-balancer/maintenance -d '{"enabled": false}'
   ```

2. **Verify health endpoints**
   ```bash
   curl http://localhost:3000/health
   ```

3. **Monitor metrics**
   - Open Grafana dashboards
   - Check error rates
   - Check response times

4. **Verify critical flows**
   - Create workflow
   - Execute workflow
   - Query data

### Phase 5: Stabilization (T+30 minutes)

1. **Monitor for 30 minutes**
   - Error rate < 1%
   - Response time p95 < 500ms
   - No critical alerts

2. **Scale up if needed**
   ```bash
   docker-compose up -d --scale bubblelab-api=3
   ```

3. **Notify team**
   ```bash
   # Post to Slack
   "@channel Deployment successful. System operating normally."
   ```

---

## Post-Deployment Verification

### 1. Health Checks ✅

- [ ] **API health endpoint**
  ```bash
  curl http://localhost:3000/health
  # Expected: 200 OK, {"status": "healthy"}
  ```

- [ ] **Service bubble health**
  ```bash
  curl http://localhost:3000/api/bubbles/health
  # Expected: 200 OK, all services healthy
  ```

- [ ] **Database health**
  ```bash
  curl http://localhost:3000/api/database/health
  # Expected: 200 OK, database reachable
  ```

- [ ] **Dependencies health**
  ```bash
  # Qdrant
  curl http://qdrant:6333/healthz

  # Elasticsearch
  curl http://elasticsearch:9200/_cluster/health

  # Redis
  redis-cli ping
  ```

### 2. Functional Tests ✅

- [ ] **User authentication**
  - [ ] Login works
  - [ ] Token refresh works
  - [ ] Logout works

- [ ] **Workflow operations**
  - [ ] Create workflow
  - [ ] Execute workflow
  - [ ] Get workflow status
  - [ ] Delete workflow

- [ ] **Service bubbles**
  - [ ] Qdrant operations work
  - [ ] Elasticsearch operations work
  - [ ] Redis operations work
  - [ ] PostgreSQL operations work

- [ ] **API endpoints**
  - [ ] GET endpoints work
  - [ ] POST endpoints work
  - [ ] PUT endpoints work
  - [ ] DELETE endpoints work

### 3. Performance Verification ✅

- [ ] **Response times acceptable**
  - [ ] p50 < 100ms ✅
  - [ ] p95 < 500ms ✅
  - [ ] p99 < 1000ms ✅

- [ ] **Error rate low**
  - [ ] Error rate < 1% ✅
  - [ ] No 5xx errors ✅

- [ ] **Throughput target met**
  - [ ] > 1000 req/min ✅

- [ ] **Resource usage normal**
  - [ ] CPU < 70% ✅
  - [ ] Memory < 70% ✅
  - [ ] Connections < 80% of pool ✅

### 4. Monitoring Verification ✅

- [ ] **Prometheus metrics**
  - [ ] All targets up
  - [ ] Metrics being collected
  - [ ] No scrape errors

- [ ] **Grafana dashboards**
  - [ ] Dashboards loading
  - [ ] Data displaying correctly
  - [ ] No data gaps

- [ ] **Alerts**
  - [ ] No unexpected alerts
  - [ ] Alert routing working
  - [ ] PagerDuty receiving alerts (if any)

- [ ] **Logs**
  - [ ] Logs being collected
  - [ ] No error spikes
  - [ ] Correlation IDs present

### 5. Data Verification ✅

- [ ] **Data integrity**
  - [ ] No data corruption
  - [ ] No data loss
  - [ ] Data migrated correctly

- [ ] **Data consistency**
  - [ ] Database consistent
  - [ ] Cache consistent
  - [ ] Search index consistent

- [ ] **Data backup**
  - [ ] Post-deployment backup created
  - [ ] Backup verified

---

## Rollback Procedures

### Rollback Decision Matrix

| Condition | Rollback Immediately | Rollback After Investigation |
|-----------|---------------------|-------------------------------|
| Error rate > 10% | ✅ | |
| Error rate > 5% for 10 min | ✅ | |
| Response time p95 > 10s | ✅ | |
| Data corruption detected | ✅ | |
| Critical services down | ✅ | |
| Error rate > 1% for 30 min | | ✅ |
| Elevated memory usage | | ✅ |
| Minor bugs | | ✅ |

### Rollback Steps

**Option 1: Instant Rollback (Docker)**

```bash
# Time: < 5 minutes

# 1. Stop current version
docker-compose stop bubblelab-api

# 2. Start previous version
docker-compose up -d bubblelab-api:previous

# 3. Verify health
curl http://localhost:3000/health

# 4. Monitor metrics
# Open Grafana dashboards
```

**Option 2: Database Rollback**

```bash
# Time: < 30 minutes

# 1. Stop application
docker-compose stop bubblelab-api

# 2. Rollback migrations
npm run migrate:down

# 3. Verify schema
npm run migrate:status

# 4. Restore data (if needed)
./scripts/restore-postgresql.sh

# 5. Start application
docker-compose start bubblelab-api

# 6. Verify health
curl http://localhost:3000/health
```

**Option 3: Full System Rollback**

```bash
# Time: < 1 hour

# 1. Stop all services
docker-compose down

# 2. Restore all backups
./scripts/restore-all.sh

# 3. Start all services
docker-compose up -d

# 4. Verify all services
./scripts/verify-all-services.sh
```

### Rollback Verification

- [ ] **Health checks pass**
- [ ] **Error rate normal**
- [ ] **Response times normal**
- [ ] **Data integrity verified**
- [ ] **No data loss**
- [ ] **Monitoring shows normal**

### Post-Rollback Actions

1. **Notify team**
   ```bash
   # Post to Slack
   "@channel Rollback completed. Investigating issue."
   ```

2. **Create incident ticket**
   - Document rollback
   - Document issue
   - Assign owner

3. **Post-mortem**
   - Schedule meeting within 24 hours
   - Document root cause
   - Create action items

---

## Health Check Endpoints

### API Health Endpoints

#### Root Health Endpoint

```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-18T14:30:00Z",
  "version": "1.0.0",
  "dependencies": {
    "postgresql": "healthy",
    "qdrant": "healthy",
    "elasticsearch": "healthy",
    "redis": "healthy"
  }
}
```

#### Detailed Health Endpoint

```bash
GET /health/detailed
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-18T14:30:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "dependencies": {
    "postgresql": {
      "status": "healthy",
      "latency_ms": 5,
      "connections": 10,
      "max_connections": 100
    },
    "qdrant": {
      "status": "healthy",
      "latency_ms": 10,
      "collections": 5
    },
    "elasticsearch": {
      "status": "healthy",
      "latency_ms": 15,
      "cluster_status": "green"
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 2,
      "memory_usage": "512MB",
      "connected_clients": 5
    }
  }
}
```

#### Service Bubble Health

```bash
GET /api/bubbles/health
```

**Response**:
```json
{
  "qdrant": {
    "status": "healthy",
    "circuit_breaker": "closed",
    "last_check": "2026-01-18T14:30:00Z"
  },
  "elasticsearch": {
    "status": "healthy",
    "circuit_breaker": "closed",
    "last_check": "2026-01-18T14:30:00Z"
  },
  "redis": {
    "status": "healthy",
    "circuit_breaker": "closed",
    "last_check": "2026-01-18T14:30:00Z"
  },
  "postgresql": {
    "status": "healthy",
    "circuit_breaker": "closed",
    "last_check": "2026-01-18T14:30:00Z"
  }
}
```

### Database Health Checks

#### PostgreSQL

```bash
GET /api/database/health
```

**Response**:
```json
{
  "status": "healthy",
  "latency_ms": 5,
  "connections": {
    "active": 10,
    "idle": 5,
    "max": 100
  },
  "database_size_mb": 1024,
  "last_backup": "2026-01-18T14:00:00Z"
}
```

#### Qdrant

```bash
GET /api/qdrant/health
```

**Response**:
```json
{
  "status": "healthy",
  "collections": 5,
  "vectors": 1000000,
  "memory_usage_mb": 512
}
```

#### Elasticsearch

```bash
GET /api/elasticsearch/health
```

**Response**:
```json
{
  "status": "healthy",
  "cluster_status": "green",
  "nodes": 3,
  "indices": 10,
  "documents": 500000
}
```

#### Redis

```bash
GET /api/redis/health
```

**Response**:
```json
{
  "status": "healthy",
  "connected_clients": 5,
  "used_memory_mb": 256,
  "uptime_seconds": 3600
}
```

### Liveness Probe

```bash
GET /health/live
```

**Purpose**: Check if container should be restarted

**Response**:
```json
{
  "status": "alive"
}
```

**Kubernetes Configuration**:
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 3000
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3
```

### Readiness Probe

```bash
GET /health/ready
```

**Purpose**: Check if container is ready to receive traffic

**Response**:
```json
{
  "status": "ready",
  "dependencies_ready": true
}
```

**Kubernetes Configuration**:
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 3000
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 3
```

---

## Graceful Shutdown Procedures

### Why Graceful Shutdown?

Graceful shutdown ensures:
1. In-flight requests complete
2. No data loss
3. Connections closed properly
4. Resources released
5. Clean state for restart

### Graceful Shutdown Implementation

#### Process 1: Docker Compose Shutdown

```bash
# Graceful shutdown (sends SIGTERM)
docker-compose stop bubblelab-api

# Wait for graceful shutdown (default 10 seconds)
# If processes don't exit, SIGKILL sent after timeout
```

#### Process 2: Kubernetes Shutdown

```yaml
# Kubernetes terminationGracePeriodSeconds
terminationGracePeriodSeconds: 30

# PreStop hook
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh", "-c", "node graceful-shutdown.js"]
```

#### Process 3: Application-Level Shutdown

```typescript
// graceful-shutdown.ts
import http from 'http';

const server = http.createServer(app);

// Graceful shutdown handler
async function gracefulShutdown(signal: string) {
  console.log(`Received ${signal}, starting graceful shutdown`);

  // 1. Stop accepting new connections
  server.close(() => {
    console.log('HTTP server closed');
  });

  // 2. Wait for in-flight requests to complete (max 30 seconds)
  const shutdownTimeout = setTimeout(() => {
    console.error('Forced shutdown after timeout');
    process.exit(1);
  }, 30000);

  try {
    // 3. Close database connections
    await database.close();
    console.log('Database connections closed');

    // 4. Close Redis connection
    await redis.quit();
    console.log('Redis connection closed');

    // 5. Close other service connections
    await qdrant.close();
    await elasticsearch.close();
    console.log('Service connections closed');

    // 6. Clear shutdown timeout
    clearTimeout(shutdownTimeout);

    // 7. Exit successfully
    console.log('Graceful shutdown complete');
    process.exit(0);
  } catch (error) {
    console.error('Error during graceful shutdown:', error);
    process.exit(1);
  }
}

// Listen for shutdown signals
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
```

### Graceful Shutdown Checklist

#### Pre-Shutdown

- [ ] **Drain connections** (load balancer)
  ```bash
  # Remove from load balancer rotation
  kubectl drain node-name --ignore-daemonsets --delete-emptydir-data
  ```

- [ ] **Wait for in-flight requests**
  - Check active connections
  - Estimate completion time
  - Set shutdown timeout accordingly

- [ ] **Stop accepting new work**
  - Set maintenance mode
  - Unregister from service discovery
  - Scale down load balancer

#### Shutdown Process

- [ ] **Send SIGTERM to process**
  ```bash
  kill -SIGTERM <pid>
  ```

- [ ] **Wait for graceful shutdown**
  - Monitor logs for "Graceful shutdown complete"
  - Wait up to 30 seconds

- [ ] **Verify all connections closed**
  - Check database connections: `pg_stat_activity`
  - Check Redis connections: `CLIENT LIST`
  - Check HTTP connections: `netstat`

- [ ] **Force kill if needed**
  ```bash
  # If graceful shutdown fails after 30s
  kill -SIGKILL <pid>
  ```

#### Post-Shutdown

- [ ] **Verify clean state**
  - No orphaned processes
  - No open connections
  - No locked resources

- [ ] **Clean up resources**
  - Remove from monitoring
  - Clean up temporary files
  - Release IP addresses

### Zero-Downtime Deployment

#### Blue-Green Deployment

```bash
# 1. Deploy new version to green environment
docker-compose -f docker-compose.green.yml up -d

# 2. Verify green environment
curl http://green.example.com/health

# 3. Switch traffic to green
# Update load balancer

# 4. Monitor green environment
# If issues, switch back to blue

# 5. Deprovision blue environment
docker-compose -f docker-compose.blue.yml down
```

#### Rolling Update

```bash
# Kubernetes rolling update
kubectl set image deployment/bubblelab-api \
  bubblelab-api=bubblelab-api:v2.0.0 \
  --namespace=production

# Monitor rollout
kubectl rollout status deployment/bubblelab-api \
  --namespace=production

# Rollback if needed
kubectl rollout undo deployment/bubblelab-api \
  --namespace=production
```

#### Canary Deployment

```bash
# 1. Deploy canary (10% of traffic)
kubectl apply -f canary-deployment.yml

# 2. Monitor canary metrics
# - Error rate
# - Response time
# - Resource usage

# 3. If canary healthy, rollout to 50%
kubectl scale deployment bubblelab-api --replicas=5

# 4. If still healthy, full rollout
kubectl scale deployment bubblelab-api --replicas=10

# 5. Remove canary
kubectl delete deployment bubblelab-api-canary
```

---

## Deployment Summary

### Deployment Record

**Deployment Date**: _______________
**Deployment Time**: _______________
**Deployed By**: _______________
**Approved By**: _______________
**Version**: _______________

**Pre-Deployment Checklist**: ✅ PASS / ❌ FAIL
**Post-Deployment Verification**: ✅ PASS / ❌ FAIL
**Rollback Required**: YES / NO

**Issues Encountered**:
-
-
-

**Actions Required**:
-
-
-

**Notes**:
-

### Sign-Off

**Deployment Engineer**: _______________
**Date**: _______________
**Signature**: _______________

**Engineering Lead**: _______________
**Date**: _______________
**Signature**: _______________

**DevOps Lead**: _______________
**Date**: _______________
**Signature**: _______________

---

## Appendix: Deployment Scripts

### Pre-Deployment Script

```bash
#!/bin/bash
# scripts/pre-deployment-check.sh

echo "Running pre-deployment checks..."

# Validate configuration
echo "Validating configuration..."
node config/validate-config.js --env production --strict
if [ $? -ne 0 ]; then
  echo "Configuration validation failed"
  exit 1
fi

# Run tests
echo "Running tests..."
npm test
if [ $? -ne 0 ]; then
  echo "Tests failed"
  exit 1
fi

# Create backup
echo "Creating backup..."
./scripts/backup-all.sh

echo "Pre-deployment checks complete"
```

### Deployment Script

```bash
#!/bin/bash
# scripts/deploy.sh

VERSION=$1

if [ -z "$VERSION" ]; then
  echo "Usage: ./deploy.sh <version>"
  exit 1
fi

echo "Deploying version $VERSION..."

# Pre-deployment checks
./scripts/pre-deployment-check.sh

# Build new version
echo "Building new version..."
docker-compose build

# Deploy new version
echo "Deploying new version..."
docker-compose up -d

# Wait for startup
echo "Waiting for startup..."
sleep 30

# Run smoke tests
echo "Running smoke tests..."
npm run smoke-test

# Verify deployment
echo "Verifying deployment..."
curl http://localhost:3000/health

echo "Deployment complete"
```

### Rollback Script

```bash
#!/bin/bash
# scripts/rollback.sh

echo "Rolling back deployment..."

# Stop current version
docker-compose stop bubblelab-api

# Start previous version
docker-compose up -d bubblelab-api:previous

# Wait for startup
sleep 30

# Verify rollback
curl http://localhost:3000/health

echo "Rollback complete"
```

---

**Last Updated**: 2026-01-18
**Next Review**: 2026-02-18
**Maintained By**: DevOps Team

---

## Quick Reference

### Pre-Deployment Commands

```bash
# Validate configuration
node config/validate-config.js --env production --strict

# Run tests
npm test

# Create backup
./scripts/backup-all.sh
```

### Deployment Commands

```bash
# Deploy new version
docker-compose build
docker-compose up -d

# Run migrations
npm run migrate:up

# Verify health
curl http://localhost:3000/health
```

### Post-Deployment Commands

```bash
# Run smoke tests
npm run smoke-test

# Monitor logs
tail -f /var/log/bubblelab/app.log

# Check metrics
curl http://localhost:9090/metrics
```

### Rollback Commands

```bash
# Instant rollback
docker-compose stop bubblelab-api
docker-compose up -d bubblelab-api:previous

# Database rollback
npm run migrate:down

# Full rollback
./scripts/restore-all.sh
```

---

**Deployment Checklist Status**: ✅ COMPLETE
**Ready for Production**: YES

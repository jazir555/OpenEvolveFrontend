# LoongFlow Deployment Guide

## Overview

This guide covers deploying the LoongFlow adapter and core service in both development (Docker Compose) and production (Kubernetes) environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Local Development (Docker Compose)](#local-development-docker-compose)
- [Production (Kubernetes)](#production-kubernetes)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

---

## Prerequisites

### Required Software

- **Docker**: 20.10+ (for local development)
- **Docker Compose**: 2.0+ (for local development)
- **Kubernetes**: 1.25+ (for production)
- **kubectl**: 1.25+ (for production)
- **LLM API Credentials**: OpenAI, Anthropic, or Google API key

### Required Infrastructure

- **Redis**: For event bus and caching
- **Persistent Storage**: For checkpoints and logs
- **Monitoring**: Prometheus + Grafana (recommended)

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     LoongFlow System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  LoongFlow       │         │  LoongFlow       │         │
│  │  Adapter         │────────>│  Core Service    │         │
│  │  (Port 8040)     │         │  (Port 8050)     │         │
│  └──────────────────┘         └──────────────────┘         │
│         │                            │                       │
│         │                            │                       │
│         v                            v                       │
│  ┌──────────────────────────────────────────┐              │
│  │         Event Bus (Redis)                │              │
│  └──────────────────────────────────────────┘              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Request** → LoongFlow Adapter (port 8040)
2. **Adapter** → Normalizes request to Canonical Schema
3. **Adapter** → Sends to LoongFlow Core (port 8050)
4. **Core** → Executes workflow using LLM
5. **Core** → Returns results to Adapter
6. **Adapter** → Publishes event to Event Bus
7. **Adapter** → Returns response to client

---

## Local Development (Docker Compose)

### Step 1: Configure Environment

```bash
# Copy the environment template
cp infra/.env.loongflow.example infra/.env.loongflow

# Edit with your configuration
nano infra/.env.loongflow

# Validate environment variables
bash infra/scripts/validate-env.sh
```

**Required Variables:**
```bash
LOONGFLOW_LLM_API_KEY=sk-your-openai-api-key-here
LOONGFLOW_LLM_PROVIDER=openai
LOONGFLOW_API_URL=http://loongflow-core:8000
```

The validation script will check:
- All required variables are set
- Optional variables have correct types
- Numeric values are valid
- Service will crash immediately if required vars are missing

### Step 2: Start LoongFlow Core Service

```bash
# Create the federation network if it doesn't exist
docker network create federation-network 2>/dev/null || true

# Start LoongFlow core
docker-compose -f docker-compose.loongflow-core.yml --env-file infra/.env.loongflow up -d

# View logs
docker-compose -f docker-compose.loongflow-core.yml logs -f loongflow-core

# Check health
curl http://localhost:8050/health
```

Expected output:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-22T10:30:00.000Z",
  "service": "loongflow-core"
}
```

### Step 3: Start LoongFlow Adapter

```bash
# Start the adapter
docker-compose -f infra/docker-compose-all-adapters.yml --env-file infra/.env.loongflow up loongflow

# View logs
docker-compose -f infra/docker-compose-all-adapters.yml logs -f loongflow

# Check health
curl http://localhost:8040/health
```

Expected output:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-22T10:30:00.000Z",
  "service": "loongflow-adapter",
  "core_connection": "healthy"
}
```

### Step 4: Verify Deployment

```bash
# Check all containers
docker ps | grep loongflow

# Test workflow execution
curl -X POST http://localhost:8040/api/v1/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Prove that the sum of two even numbers is even",
    "context": {
      "domain": "mathematics",
      "language": "lean4"
    }
  }'
```

### Stopping Services

```bash
# Stop adapter
docker-compose -f infra/docker-compose-all-adapters.yml down

# Stop core service
docker-compose -f docker-compose.loongflow-core.yml down

# Remove volumes (WARNING: deletes data)
docker-compose -f docker-compose.loongflow-core.yml down -v
```

---

## Production (Kubernetes)

### Step 1: Create Namespace

```bash
# Apply the namespace
kubectl apply -f infra/k8s-loongflow-deployment.yaml

# Verify namespace creation
kubectl get namespace loongflow-system
```

### Step 2: Configure Secrets

```bash
# Validate environment variables first
source infra/scripts/validate-env.sh

# Create secret for LLM API key
kubectl create secret generic loongflow-core-secrets \
  --from-literal=LLM_API_KEY=$LOONGFLOW_LLM_API_KEY \
  -n loongflow-system

# Verify secret
kubectl get secret loongflow-core-secrets -n loongflow-system
```

## ⚠️ IMPORTANT: Secrets Management

DO NOT commit API keys to the repository.

Create secrets using:
```bash
kubectl create secret generic loongflow-core-secrets \
  --from-literal=LLM_API_KEY=$your_key \
  -n loongflow-system
```

For production, use:
- External Secrets Operator
- Sealed Secrets
- Vault
- Cloud provider secret management

### Step 3: Deploy LoongFlow Core Service

```bash
# Apply core deployment
kubectl apply -f infra/k8s-loongflow-core.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=loongflow-core -n loongflow-system --timeout=300s

# Check deployment status
kubectl get deployment loongflow-core -n loongflow-system

# View logs
kubectl logs -f deployment/loongflow-core -n loongflow-system
```

### Step 4: Deploy LoongFlow Adapter

```bash
# Apply adapter deployment
kubectl apply -f infra/k8s-loongflow-deployment.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=loongflow-adapter -n loongflow-system --timeout=300s

# Check deployment status
kubectl get deployment loongflow-adapter -n loongflow-system

# View logs
kubectl logs -f deployment/loongflow-adapter -n loongflow-system
```

### Step 5: Verify Deployment

```bash
# Check all pods
kubectl get pods -n loongflow-system

# Check services
kubectl get svc -n loongflow-system

# Check HPA status
kubectl get hpa -n loongflow-system

# Port-forward to test locally
kubectl port-forward svc/loongflow-adapter-service 8040:8040 -n loongflow-system

# Test health endpoint
curl http://localhost:8040/health
```

### Step 6: Monitor Scaling

```bash
# Check HPA status
kubectl describe hpa loongflow-adapter-hpa -n loongflow-system

# Check resource usage
kubectl top pods -n loongflow-system

# Check pod distribution
kubectl get pods -n loongflow-system -o wide
```

---

## Configuration

### Environment Variables

#### Core Service Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LOONGFLOW_LLM_PROVIDER` | Yes | - | LLM provider (openai, anthropic, google) |
| `LOONGFLOW_LLM_API_KEY` | Yes | - | LLM API key |
| `LOONGFLOW_LLM_MODEL` | No | gpt-4 | Model name |
| `LOONGFLOW_LLM_TEMPERATURE` | No | 0.7 | Temperature for generation |
| `LOONGFLOW_MAX_CONCURRENT_WORKFLOWS` | No | 10 | Max concurrent workflows |
| `LOONGFLOW_WORKFLOW_TIMEOUT_MS` | No | 300000 | Workflow timeout (5 minutes) |
| `LOONGFLOW_ENABLE_CHECKPOINTING` | No | true | Enable workflow checkpointing |
| `LOONGFLOW_REDIS_URL` | Yes | - | Redis URL for caching |
| `TZ` | Yes | UTC | Timezone (Law of UTC) |

#### Adapter Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LOONGFLOW_API_URL` | Yes | - | LoongFlow Core API URL |
| `LOONGFLOW_TIMEOUT_MS` | No | 30000 | Request timeout (30 seconds) |
| `LOONGFLOW_MAX_RETRIES` | No | 3 | Max retry attempts |
| `PORT` | No | 8040 | Adapter port |
| `LOG_LEVEL` | No | INFO | Log level |
| `TZ` | Yes | UTC | Timezone (Law of UTC) |

### Configuring LLM Providers

#### OpenAI
```bash
LOONGFLOW_LLM_PROVIDER=openai
LOONGFLOW_LLM_API_KEY=sk-...
LOONGFLOW_LLM_MODEL=gpt-4
```

#### Anthropic
```bash
LOONGFLOW_LLM_PROVIDER=anthropic
LOONGFLOW_LLM_API_KEY=sk-ant-...
LOONGFLOW_LLM_MODEL=claude-3-opus-20240229
```

#### Google
```bash
LOONGFLOW_LLM_PROVIDER=google
LOONGFLOW_LLM_API_KEY=AI...
LOONGFLOW_LLM_MODEL=gemini-pro
```

### Hybrid Orchestration Configuration

The adapter supports hybrid orchestration combining PES (Prompt Evolution Strategy) and RESE (Recursive Exploration Synthesis Engine).

```bash
# Enable hybrid mode
LOONGFLOW_ENABLE_HYBRID=true

# Enable PES fallback when RESE fails
LOONGFLOW_ENABLE_PES_FALLBACK=true

# PES fallback configuration
LOONGFLOW_PES_TIMEOUT_MS=60000
LOONGFLOW_PES_MAX_ITERATIONS=10
```

---

## Monitoring

### Health Checks

Both services expose `/health` endpoints:

```bash
# Core service
curl http://localhost:8050/health

# Adapter
curl http://localhost:8040/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-22T10:30:00.000Z",
  "service": "loongflow-adapter",
  "core_connection": "healthy",
  "metrics": {
    "active_workflows": 3,
    "completed_workflows": 127,
    "failed_workflows": 2,
    "avg_execution_time_ms": 15234
  }
}
```

### Metrics (Prometheus)

Both services expose `/metrics` endpoints for Prometheus scraping:

```bash
# Adapter metrics
curl http://localhost:8040/metrics

# Core metrics
curl http://localhost:8050/metrics
```

Key metrics:
- `loongflow_workflows_total`: Total workflows executed
- `loongflow_workflows_duration_seconds`: Workflow execution duration
- `loongflow_workflows_failed_total`: Failed workflows
- `loongflow_llm_requests_total`: LLM API requests
- `loongflow_llm_tokens_total`: LLM tokens consumed

### Logging

Logs are structured JSON for easy parsing:

```json
{
  "timestamp": "2026-02-22T10:30:00.000Z",
  "level": "INFO",
  "service": "loongflow-adapter",
  "correlation_id": "abc123",
  "message": "Workflow execution started",
  "workflow_id": "wf-456",
  "goal": "Prove theorem X"
}
```

View logs:

```bash
# Docker Compose
docker-compose -f docker-compose.loongflow-core.yml logs -f loongflow-core

# Kubernetes
kubectl logs -f deployment/loongflow-core -n loongflow-system
```

### Grafana Dashboards

Import the provided Grafana dashboard (`monitoring/grafana/dashboards/loongflow.json`) for visualizing:

- Workflow execution rate
- LLM API usage and costs
- Error rates and latency
- Resource utilization

---

## Troubleshooting

### Common Issues

#### 1. Core Service Not Starting

**Symptoms:** Container exits immediately

**Diagnosis:**
```bash
# Check logs
docker-compose -f docker-compose.loongflow-core.yml logs loongflow-core

# Look for missing required variables
# Output: "ERROR: LOONGFLOW_LLM_API_KEY is required"
```

**Solution:** Set all required environment variables in `.env.loongflow`

#### 2. Adapter Cannot Connect to Core

**Symptoms:** Health check shows `"core_connection": "unhealthy"`

**Diagnosis:**
```bash
# Check if core is running
docker ps | grep loongflow-core

# Check network connectivity
docker exec loongflow-adapter ping -c 1 loongflow-core

# Check core health
curl http://localhost:8050/health
```

**Solution:**
- Ensure core service is healthy
- Check network configuration (both on `federation-network`)
- Verify `LOONGFLOW_API_URL` is correct

#### 3. High Memory Usage

**Symptoms:** Pods OOMKilled or containers restart

**Diagnosis:**
```bash
# Check resource usage
docker stats loongflow-core

# Kubernetes
kubectl top pod -n loongflow-system
```

**Solution:**
- Increase memory limits in docker-compose or Kubernetes
- Reduce `LOONGFLOW_MAX_CONCURRENT_WORKFLOWS`
- Enable checkpointing to reduce in-memory state

#### 4. Slow Workflow Execution

**Symptoms:** Workflows take longer than expected

**Diagnosis:**
```bash
# Check logs for slow operations
docker logs loongflow-core | grep "duration"

# Check LLM API latency
kubectl logs -f deployment/loongflow-core -n loongflow-system | grep "llm_request_duration"
```

**Solution:**
- Reduce `LOONGFLOW_WORKFLOW_TIMEOUT_MS`
- Use faster LLM model
- Enable checkpointing for incremental progress
- Check network latency to LLM API

#### 5. Contract Tests Failing

**Symptoms:** Adapter refuses to start with contract validation errors

**Diagnosis:**
```bash
# Check logs
kubectl logs deployment/loongflow-adapter -n loongflow-system | grep "contract"
```

**Solution:**
- Core API may have changed
- Update adapter to match new API
- Skip tests temporarily: `SKIP_CONTRACT_TESTS=true`

### Debug Mode

Enable debug logging:

```bash
# Docker Compose
LOG_LEVEL=DEBUG docker-compose -f docker-compose.loongflow-core.yml up

# Kubernetes
kubectl patch configmap loongflow-core-config -n loongflow-system \
  --type merge -p '{"data":{"LOG_LEVEL":"DEBUG"}}'

# Restart pods to pick up changes
kubectl rollout restart deployment/loongflow-core -n loongflow-system
```

### Recovery Procedures

#### Recovering from Crash

```bash
# Check checkpoint data
kubectl exec -it deployment/loongflow-core -n loongflow-system -- ls -la /app/checkpoints

# Restart pods (checkpoints will be restored)
kubectl rollout restart deployment/loongflow-core -n loongflow-system
```

#### Clearing Deadlocked Workflows

```bash
# Access Redis
kubectl exec -it redis-pod -- redis-cli

# Clear workflow locks
DEL loongflow:locks:*

# Clear stuck workflows
DEL loongflow:workflows:stuck:*
```

---

## Maintenance

### Backup and Restore

#### Backing Up Checkpoints

```bash
# Docker Compose
docker run --rm -v loongflow-checkpoints:/data -v $(pwd):/backup \
  alpine tar czf /backup/loongflow-checkpoints-$(date +%Y%m%d).tar.gz /data

# Kubernetes
kubectl exec -n loongflow-system deployment/loongflow-core -- \
  tar czf /tmp/checkpoints.tar.gz /app/checkpoints

kubectl cp loongflow-system/$(kubectl get pods -n loongflow-system -l app=loongflow-core -o jsonpath='{.items[0].metadata.name}'):/tmp/checkpoints.tar.gz \
  ./backups/checkpoints-$(date +%Y%m%d).tar.gz
```

#### Restoring Checkpoints

```bash
# Docker Compose
docker run --rm -v loongflow-checkpoints:/data -v $(pwd):/backup \
  alpine tar xzf /backup/loongflow-checkpoints-YYYYMMDD.tar.gz -C /

# Kubernetes
kubectl cp ./backups/checkpoints-YYYYMMDD.tar.gz \
  loongflow-system/$(kubectl get pods -n loongflow-system -l app=loongflow-core -o jsonpath='{.items[0].metadata.name}'):/tmp/

kubectl exec -n loongflow-system deployment/loongflow-core -- \
  tar xzf /tmp/checkpoints-YYYYMMDD.tar.gz -C /app/
```

### Updating Deployment

#### Rolling Update (Zero Downtime)

```bash
# Build new image
docker build -t loongflow-adapter:v2.0.0 -f glue/adapters/loongflow-adapter/Dockerfile .

# Push to registry
docker tag loongflow-adapter:v2.0.0 registry.example.com/loongflow-adapter:v2.0.0
docker push registry.example.com/loongflow-adapter:v2.0.0

# Update Kubernetes deployment
kubectl set image deployment/loongflow-adapter \
  loongflow-adapter=registry.example.com/loongflow-adapter:v2.0.0 \
  -n loongflow-system

# Monitor rollout
kubectl rollout status deployment/loongflow-adapter -n loongflow-system
```

#### Rollback

```bash
# Check rollout history
kubectl rollout history deployment/loongflow-adapter -n loongflow-system

# Rollback to previous version
kubectl rollout undo deployment/loongflow-adapter -n loongflow-system

# Rollback to specific revision
kubectl rollout undo deployment/loongflow-adapter --to-revision=2 -n loongflow-system
```

### Scaling

#### Manual Scaling

```bash
# Scale adapter to 5 replicas
kubectl scale deployment/loongflow-adapter --replicas=5 -n loongflow-system

# Scale core service
kubectl scale deployment/loongflow-core --replicas=5 -n loongflow-system
```

#### Auto-Scaling (HPA)

HPA is configured by default with:
- Min replicas: 3
- Max replicas: 10
- Target CPU utilization: 70%
- Target memory utilization: 80%

Adjust HPA:
```bash
kubectl edit hpa loongflow-adapter-hpa -n loongflow-system
```

### Resource Tuning

Based on workload, adjust resource limits:

```yaml
# Low traffic (development)
resources:
  requests:
    memory: "128Mi"
    cpu: "100m"
  limits:
    memory: "256Mi"
    cpu: "250m"

# Medium traffic (staging)
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"

# High traffic (production)
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

---

## Security Considerations

### API Keys

- **NEVER** commit API keys to version control
- Use Kubernetes Secrets or Docker Secrets in production
- Rotate keys regularly
- Use separate keys for dev/staging/prod

### Network Security

- Use NetworkPolicies to restrict traffic
- Enable TLS for external endpoints
- Use service mesh (Istio, Linkerd) for mTLS

### RBAC

Create minimal RBAC roles:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: loongflow-adapter
  namespace: loongflow-system
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get"]
```

---

## Performance Optimization

### Checklist

- [ ] Enable checkpointing for long-running workflows
- [ ] Configure appropriate timeouts
- [ ] Use HPA for auto-scaling
- [ ] Enable PES cost optimization
- [ ] Monitor LLM API costs
- [ ] Use caching where possible
- [ ] Tune resource limits based on usage

### Tips

1. **Batch Workflows**: Execute multiple workflows in parallel when possible
2. **Use Checkpointing**: Reduces recomputation on failures
3. **Choose Right Model**: Use smaller models for simple tasks
4. **Monitor Costs**: Set budget limits for PES evolution
5. **Optimize Prompts**: Reduce token usage and improve response quality

---

## Support

### Documentation

- [LoongFlow Architecture](../glue/adapters/loongflow-adapter/ADR.md)
- [PES Integration](../docs/pes-integration.md)
- [Canonical Schemas](../glue/schemas/loongflow-canonical.ts)

### Logs and Debugging

For issues, collect:
1. Service logs
2. Health check outputs
3. Configuration (redacted)
4. Metrics snapshots

### Getting Help

- Check existing issues on GitHub
- Review ADR (Architecture Decision Records)
- Consult troubleshooting section above

---

**Last Updated:** 2026-02-22
**Version:** 1.0.0

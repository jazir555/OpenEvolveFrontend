# LoongFlow Adapter Deployment Reference

## Quick Start

### Local Development

```bash
# 1. Configure environment
cp infra/.env.loongflow.example infra/.env.loongflow
nano infra/.env.loongflow  # Set LOONGFLOW_LLM_API_KEY

# 2. Deploy
./infra/scripts/deploy-loongflow.sh local

# 3. Validate
./infra/scripts/validate-loongflow-deployment.sh local

# 4. Test
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

### Production (Kubernetes)

```bash
# 1. Configure secrets
kubectl create secret generic loongflow-core-secrets \
  --from-literal=LLM_API_KEY=sk-your-key \
  -n loongflow-system

# 2. Deploy
kubectl apply -f infra/k8s-loongflow-core.yaml
kubectl apply -f infra/k8s-loongflow-deployment.yaml

# 3. Verify
kubectl get pods -n loongflow-system
kubectl get svc -n loongflow-system
```

## Architecture

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

## Files Created

| File | Purpose |
|------|---------|
| `docker-compose.loongflow-core.yml` | Core service Docker Compose |
| `infra/docker-compose-all-adapters.yml` | Updated with LoongFlow adapter |
| `infra/k8s-loongflow-deployment.yaml` | Kubernetes adapter deployment |
| `infra/k8s-loongflow-core.yaml` | Kubernetes core deployment |
| `infra/.env.loongflow.example` | Environment template |
| `infra/LOONGFLOW_DEPLOYMENT.md` | Full deployment guide |
| `infra/scripts/deploy-loongflow.sh` | Quick deployment script |
| `infra/scripts/validate-loongflow-deployment.sh` | Validation script |

## Environment Variables

### Required

```bash
LOONGFLOW_LLM_API_KEY=sk-...           # LLM API key
LOONGFLOW_API_URL=http://loongflow-core:8050  # Core URL
```

### Optional

```bash
LOONGFLOW_PORT=8040                    # Adapter port
LOONGFLOW_CORE_PORT=8050               # Core port
LOONGFLOW_TIMEOUT_MS=30000             # Request timeout
LOONGFLOW_MAX_RETRIES=3                # Retry attempts
LOG_LEVEL=INFO                         # Logging level
TZ=UTC                                 # Timezone (required)
```

## Health Checks

```bash
# Core service
curl http://localhost:8050/health

# Adapter
curl http://localhost:8040/health

# Metrics
curl http://localhost:8040/metrics
```

## Monitoring

All services expose Prometheus metrics at `/metrics`:

- `loongflow_workflows_total`
- `loongflow_workflows_duration_seconds`
- `loongflow_llm_requests_total`
- `loongflow_llm_tokens_total`

## Troubleshooting

See [LOONGFLOW_DEPLOYMENT.md](../../infra/LOONGFLOW_DEPLOYMENT.md) for full troubleshooting guide.

## Laws Followed

- ✅ **Law of Configuration Explicitness**: All config via env vars
- ✅ **Law of Runtime Truth**: Health checks verify real API availability
- ✅ **Law of UTC**: All services use TZ=UTC
- ✅ **Law of Idempotency**: Safe to run deployment multiple times

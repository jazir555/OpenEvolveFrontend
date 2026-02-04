# RESE Pipeline Deployment Guide

This directory contains deployment manifests for the RESE Pipeline Orchestrator.

## Deployment Options

### 1. Docker Compose (Local Development)

**Best for**: Local development, testing, single-host deployment

**Prerequisites:**
- Docker Desktop installed
- 8GB+ RAM available

**Quick Start:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f rese-pipeline

# Execute pipeline
docker-compose exec rese-pipeline python -m rese_pipeline --problem "Test problem"

# Stop services
docker-compose down
```

**Configuration:**
- Edit `docker-compose.yml` to change environment variables
- All services share `rese-network` Docker network
- Data persisted in `rese-data` volume

**Services:**
- `rese-pipeline`: Main orchestrator (port 8000)
- `rese-dee`: Deep Exploration Engine (port 8001)
- `rese-lltl`: Logic-to-Loss Translation (port 8002)
- `rese-sce`: Symbolic Constraint Engine (port 8003)

---

### 2. Kubernetes (Production)

**Best for**: Production, multi-node, scalable deployment

**Prerequisites:**
- Kubernetes cluster (v1.19+)
- kubectl configured
- 3+ nodes, 16GB+ RAM total

**Quick Start:**
```bash
# Deploy to Kubernetes
kubectl apply -f k8s-rese-deployment.yaml

# Check deployment status
kubectl get pods -n rese-system

# View logs
kubectl logs -f deployment/rese-pipeline -n rese-system

# Get service URL
kubectl get svc rese-pipeline-service -n rese-system
```

**Architecture:**
- Namespace: `rese-system`
- ConfigMap: `rese-pipeline-config`
- Deployments: `rese-pipeline`, `rese-dee`, `rese-lltl`, `rese-sce`
- Services: ClusterIP (internal)
- Ingress: Optional (for external access)

**Scaling:**
```bash
# Scale replicas
kubectl scale deployment rese-pipeline --replicas=5 -n rese-system

# Configure autoscaling
kubectl autoscale deployment rese-pipeline \
  --min=3 --max=10 \
  --cpu-percent=70 \
  -n rese-system
```

**Monitoring:**
```bash
# Get pod status
kubectl get pods -n rese-system -w

# Describe pod (for debugging)
kubectl describe pod <pod-name> -n rese-system

# Port forward to local
kubectl port-forward svc/rese-pipeline-service 8000:8000 -n rese-system
```

---

## Configuration Management

### Environment Variables

All configuration is via environment variables. See `glue/orchestration/config.py` for the complete list.

### Docker Compose

Edit `docker-compose.yml`:

```yaml
environment:
  - PIPELINE_TIMEOUT_MS=300000
  - PHASE_I_TIMEOUT_MS=60000
  # ... (see full list in docker-compose.yml)
```

### Kubernetes

Edit `k8s-rese-deployment.yaml` ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rese-pipeline-config
data:
  PIPELINE_TIMEOUT_MS: "300000"
  PHASE_I_TIMEOUT_MS: "60000"
  # ... (see full list in k8s-rese-deployment.yaml)
```

Or use `kubectl edit configmap rese-pipeline-config -n rese-system`:

```bash
kubectl edit configmap rese-pipeline-config -n rese-system

# Restart pods to pick up changes
kubectl rollout restart deployment rese-pipeline -n rese-system
```

---

## Storage

### Docker Compose

Uses named volume `rese-data`:

```yaml
volumes:
  rese-data:
    driver: local
```

**Location:**
- Linux: `/var/lib/docker/volumes/rese-data/_data`
- Mac/Windows: In Docker VM

**Backup:**
```bash
docker run --rm -v rese-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/rese-data-backup.tar.gz /data
```

**Restore:**
```bash
docker run --rm -v rese-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/rese-data-backup.tar.gz -C /
```

### Kubernetes

Uses PersistentVolumeClaim `rese-data-pvc`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rese-data-pvc
  namespace: rese-system
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

**Storage Classes:**
- `standard`: Default cluster storage class
- Use fast storage (SSD) for production

**Backup:**
```bash
# Get PVC name
kubectl get pvc -n rese-system

# Create backup
kubectl exec -n rese-system <pod-name> -- tar czf /tmp/backup.tar.gz /data
kubectl cp rese-system/<pod-name>:/tmp/backup.tar.gz ./backup.tar.gz
```

---

## Health Checks

### Docker Compose

```bash
# Check health status
docker-compose ps

# View health check logs
docker-compose logs rese-pipeline | grep health
```

Health check endpoints:
- `http://localhost:8000/health` - Liveness
- `http://localhost:8000/ready` - Readiness

### Kubernetes

```bash
# Check pod health
kubectl get pods -n rese-system

# Describe pod for health check status
kubectl describe pod <pod-name> -n rese-system
```

Probes configured in deployment:
- Liveness: `/health` (every 10s)
- Readiness: `/ready` (every 5s)

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs rese-pipeline
# or
kubectl logs <pod-name> -n rese-system

# Common issues:
# 1. Missing environment variables → Check configuration
# 2. Port already in use → Change port mapping
# 3. Volume mount error → Check volume paths
```

### Pipeline failing

```bash
# View detailed logs
docker-compose logs -f rese-pipeline
# or
kubectl logs -f deployment/rese-pipeline -n rese-system

# Check for correlation_id
# All logs have correlation_id for tracing

# Check Dead Letter Queue
curl http://localhost:8000/api/v1/dlq
```

### Circuit Breaker Open

```bash
# Check circuit breaker status
curl http://localhost:8000/api/v1/stats/circuit-breaker

# Reset circuit breaker (if service recovered)
curl -X POST http://localhost:8000/api/v1/circuit-breaker/reset
```

### Out of Memory

```bash
# Docker Compose: Edit memory limits in docker-compose.yml
# Kubernetes: Edit resource limits in deployment

resources:
  limits:
    memory: "4Gi"  # Increase as needed
```

---

## Monitoring

### Prometheus (Optional)

Uncomment in `docker-compose.yml`:

```yaml
prometheus:
  image: prom/prometheus:latest
  ports:
    - "9090:9090"
```

Access at `http://localhost:9090`

### Grafana (Optional)

Uncomment in `docker-compose.yml`:

```yaml
grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
```

Access at `http://localhost:3000` (admin/admin)

### Metrics

Pipeline exposes metrics at `/metrics` (Prometheus format):

- `rese_pipeline_executions_total`
- `rese_pipeline_duration_seconds`
- `rese_phase_duration_seconds`
- `rese_circuit_breaker_state`
- `rese_dead_letter_queue_size`

---

## Security

### Secrets Management

**Docker Compose:**
```bash
# Use .env file (not in git)
echo "API_KEY=secret" > .env
echo ".env" >> .gitignore

# Reference in docker-compose.yml:
environment:
  - API_KEY=${API_KEY}
```

**Kubernetes:**
```bash
# Create secret
kubectl create secret generic rese-secrets \
  --from-literal=API_KEY=secret \
  -n rese-system

# Reference in deployment:
env:
  - name: API_KEY
    valueFrom:
      secretKeyRef:
        name: rese-secrets
        key: API_KEY
```

### Network Policies (Kubernetes)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rese-network-policy
  namespace: rese-system
spec:
  podSelector:
    matchLabels:
      app: rese-pipeline
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector: {}
  egress:
  - to:
    - namespaceSelector: {}
```

---

## Performance Tuning

### Resource Limits

**Development:**
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

**Production:**
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Timeout Tuning

Adjust based on workload:

```bash
# For complex problems (longer MCTS)
export PHASE_III_TIMEOUT_MS=300000  # 5 minutes
export PIPELINE_TIMEOUT_MS=600000   # 10 minutes

# For simple problems (faster execution)
export PHASE_III_TIMEOUT_MS=60000   # 1 minute
export PIPELINE_TIMEOUT_MS=180000   # 3 minutes
```

### Parallel Execution

Enable multiple instances:

```bash
# Docker Compose: Scale services
docker-compose up -d --scale rese-pipeline=3

# Kubernetes: Use HPA
kubectl autoscale deployment rese-pipeline \
  --min=3 --max=10 --cpu-percent=70 \
  -n rese-system
```

---

## Upgrade Strategy

### Rolling Update (Kubernetes)

```bash
# Build new image
docker build -t rese-pipeline:v2 .

# Update deployment image
kubectl set image deployment/rese-pipeline \
  rese-pipeline=rese-pipeline:v2 \
  -n rese-system

# Monitor rollout
kubectl rollout status deployment/rese-pipeline -n rese-system

# Rollback if needed
kubectl rollout undo deployment/rese-pipeline -n rese-system
```

### Blue-Green Deployment

```bash
# Deploy new version
kubectl apply -f k8s-rese-deployment-v2.yaml

# Switch traffic
kubectl patch svc rese-pipeline-service \
  -p '{"spec":{"selector":{"version":"v2"}}}' \
  -n rese-system

# Rollback if needed
kubectl patch svc rese-pipeline-service \
  -p '{"spec":{"selector":{"version":"v1"}}}' \
  -n rese-system
```

---

## Disaster Recovery

### Backup Procedure

1. **Backup data:**
```bash
kubectl exec -n rese-system <pod-name> -- tar czf /tmp/backup.tar.gz /data
kubectl cp rese-system/<pod-name>:/tmp/backup.tar.gz ./backup-$(date +%Y%m%d).tar.gz
```

2. **Backup configuration:**
```bash
kubectl get configmap rese-pipeline-config -n rese-system -o yaml > config-backup.yaml
kubectl get secret rese-secrets -n rese-system -o yaml > secrets-backup.yaml
```

3. **Backup deployment:**
```bash
kubectl get deployment rese-pipeline -n rese-system -o yaml > deployment-backup.yaml
```

### Restore Procedure

1. **Restore data:**
```bash
kubectl cp ./backup.tar.gz rese-system/<pod-name>:/tmp/restore.tar.gz
kubectl exec -n rese-system <pod-name> -- tar xzf /tmp/restore.tar.gz -C /
```

2. **Restore configuration:**
```bash
kubectl apply -f config-backup.yaml
kubectl apply -f secrets-backup.yaml
```

3. **Restart deployment:**
```bash
kubectl rollout restart deployment rese-pipeline -n rese-system
```

---

## Support

For issues or questions:

1. Check logs for correlation_id
2. Review ADR.md for architecture decisions
3. See README.md for usage examples
4. Check Dead Letter Queue for failed operations

## License

MIT

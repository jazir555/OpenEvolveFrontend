# OpenEvolve Knowledge Engine - Production Deployment Guide

**Version:** 2.0.0  
**Date:** 2026-02-03  
**Status:** Production Ready

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd knowledge_engine

# Install dependencies
pip install -r requirements.txt

# For production API
pip install fastapi uvicorn

# For cloud storage
pip install boto3 google-cloud-storage azure-storage-blob

# For real embeddings
pip install sentence-transformers

# For monitoring
pip install psutil
```

### 2. Environment Variables

Create `.env` file:

```bash
# API Configuration
KE_API_KEY=your-secure-api-key-here

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Storage Configuration
STORAGE_BACKEND=memory  # memory, postgresql, qdrant
STORAGE_PATH=./data

# Cloud Storage (optional)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_BUCKET_NAME=your-bucket

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=info
```

### 3. Run Health Check

```bash
python -c "
from knowledge_engine import quick_health_check
import asyncio
result = asyncio.run(quick_health_check())
print(result)
"
```

### 4. Start Production Server

```bash
# Using Python
python -m knowledge_engine.production_api

# Or using uvicorn directly
uvicorn knowledge_engine.production_api:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### Health & Monitoring

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | System health status |
| `/ready` | GET | No | Ready check for Kubernetes |
| `/metrics` | GET | Yes | Prometheus metrics |
| `/config` | GET | Yes | System configuration |

### Core Services

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/embed` | POST | Yes | Generate embeddings |
| `/confidence` | POST | Yes | Calculate confidence scores |
| `/strategy` | POST | Yes | Recommend processing strategy |

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY knowledge_engine/ ./knowledge_engine/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from knowledge_engine import quick_health_check; import asyncio; asyncio.run(quick_health_check())" || exit 1

# Run server
CMD ["uvicorn", "knowledge_engine.production_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  knowledge-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - KE_API_KEY=${KE_API_KEY}
      - STORAGE_BACKEND=memory
    volumes:
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Kubernetes Deployment

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: knowledge-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: knowledge-engine
  template:
    metadata:
      labels:
        app: knowledge-engine
    spec:
      containers:
      - name: knowledge-engine
        image: knowledge-engine:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: KE_API_KEY
          valueFrom:
            secretKeyRef:
              name: ke-secrets
              key: api-key
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
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: knowledge-engine
spec:
  selector:
    app: knowledge-engine
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Monitoring & Alerting

### Prometheus Metrics

The `/metrics` endpoint provides:

```
ke_health{status="healthy"} 1
ke_component_health{component="embedding_service"} 1
ke_component_latency_ms{component="embedding_service"} 45.2
```

### Grafana Dashboard

Create dashboard with panels for:
- Overall health status
- Component health (embedding, confidence, strategy)
- Response latencies
- Request rates
- Error rates

### Alerts

```yaml
# High latency alert
- alert: HighEmbeddingLatency
  expr: ke_component_latency_ms{component="embedding_service"} > 1000
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Embedding service latency is high"

# Unhealthy component alert
- alert: UnhealthyComponent
  expr: ke_component_health < 1
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Knowledge Engine component is unhealthy"
```

---

## Scaling Guidelines

### Horizontal Scaling

- Scale based on request rate (> 100 req/s per pod)
- Scale based on latency (p95 > 500ms)
- Minimum 2 pods for high availability

### Resource Requirements

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| Base | 500m | 512Mi | Minimum |
| Embeddings | +500m | +1Gi | If using sentence-transformers |
| API Server | 200m | 256Mi | Per instance |

### Vertical Scaling

- Increase memory for larger embedding models
- Increase CPU for higher throughput
- Use dedicated nodes for GPU acceleration (optional)

---

## Security Best Practices

1. **API Keys**
   - Use strong, random API keys
   - Rotate keys regularly
   - Store in secure vault (not in code)

2. **Network Security**
   - Use HTTPS/TLS
   - Restrict access with firewall rules
   - Use VPC for cloud deployments

3. **Data Protection**
   - Encrypt data at rest
   - Use secure connections for storage
   - Implement audit logging

4. **Container Security**
   - Run as non-root user
   - Use minimal base images
   - Scan for vulnerabilities

---

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   
   # Verify installation
   pip list | grep knowledge-engine
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   python -c "
   from knowledge_engine.health_monitor import check_memory_usage
   print(check_memory_usage().to_dict())
   "
   ```

3. **Performance Issues**
   ```bash
   # Check component latencies
   python -c "
   from knowledge_engine import quick_health_check
   import asyncio
   result = asyncio.run(quick_health_check())
   for c in result['components']:
       print(f\"{c['name']}: {c['latency_ms']:.2f}ms\")
   "
   ```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=debug python -m knowledge_engine.production_api
```

---

## Support

For issues and questions:
- Check documentation in `docs/`
- Review logs at `/var/log/knowledge-engine/`
- Run health checks: `python -m knowledge_engine.health_monitor`

---

## License

All components use permissive open-source licenses (MIT, Apache 2.0, BSD).

**Production Ready: 100%**

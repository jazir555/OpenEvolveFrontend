# RESE Deployment Guide

Complete guide for deploying the RESE (Recursive Epistemic Solvability Engine) system in various environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Development Setup](#development-setup)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Configuration](#configuration)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Disk: 20 GB
- OS: Linux, macOS, or Windows

**Recommended**:
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 50+ GB SSD
- OS: Ubuntu 22.04 LTS or equivalent

### Software Requirements

- **Python**: 3.9 or higher
- **pip**: Latest version
- **git**: For cloning repository

### Optional Dependencies

- **Docker**: 20.10+ (for containerized deployment)
- **Nginx**: 1.18+ (for reverse proxy)
- **Redis**: 6.0+ (for caching, optional)
- **PostgreSQL**: 13+ (for persistent storage, optional)

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/rese.git
cd rese
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: API server dependencies
pip install fastapi uvicorn[standard] websockets

# Optional: Monitoring dependencies
pip install psutil prometheus-client
```

### 4. Configure RESE

```bash
# Create default configuration
python -c "from rese.config import create_default_config; create_default_config()"

# Edit configuration as needed
nano config.json
```

### 5. Run RESE Pipeline

```bash
# Run example
python -c "
from rese.rese_pipeline import run_rese

result = run_rese(
    problem_description='Test optimization',
    constraints=[{
        'id': 'c1',
        'type': 'hard',
        'description': 'Test constraint',
        'formalization': 'x > 0',
        'source': 'test'
    }],
    variables={'x': {'type': 'real'}}
)

print(f'Status: {result.status}')
print(f'Validation Score: {result.validation_score}')
"
```

### 6. Start API Server (Optional)

```bash
# Start API server
python -m rese.api
```

API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

---

## Development Setup

### Development Configuration

```python
# config/development.json
{
  "environment": "development",
  "api": {
    "debug": true,
    "host": "localhost",
    "port": 8000,
    "workers": 1
  },
  "monitoring": {
    "log_level": "DEBUG",
    "enable_profiling": true
  },
  "pipeline": {
    "enable_caching": false
  }
}
```

### Running Tests

```bash
# Run all tests
pytest rese/tests/

# Run integration tests only
pytest rese/tests/test_integration.py -v

# Run with coverage
pytest rese/tests/ --cov=rese --cov-report=html
```

### Development Server

```bash
# Start with auto-reload
uvicorn rese.api:create_app --reload --host localhost --port 8000
```

---

## Production Deployment

### System Preparation

#### 1. Create Dedicated User

```bash
sudo useradd -r -s /bin/bash rese
sudo mkdir -p /opt/rese
sudo chown rese:rese /opt/rese
```

#### 2. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.9 python3-pip python3-venv

# RHEL/CentOS
sudo yum install -y python39 python39-pip
```

### Application Setup

#### 1. Deploy Application Code

```bash
# Copy application files
sudo cp -r rese/ /opt/rese/
sudo chown -R rese:rese /opt/rese

# Create virtual environment
cd /opt/rese
sudo -u rese python3 -m venv venv
sudo -u rese venv/bin/pip install -r requirements.txt
```

#### 2. Production Configuration

Create `/opt/rese/config/production.json`:

```json
{
  "environment": "production",
  "api": {
    "debug": false,
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "enable_auth": true,
    "rate_limit_per_minute": 60
  },
  "monitoring": {
    "log_level": "INFO",
    "log_file": "/var/log/rese/rese.log",
    "enable_metrics": true,
    "metrics_port": 9090
  },
  "pipeline": {
    "enable_caching": true,
    "cache_ttl_seconds": 3600,
    "max_time_seconds": 86400
  }
}
```

#### 3. Setup Logging

```bash
# Create log directory
sudo mkdir -p /var/log/rese
sudo chown rese:rese /var/log/rese

# Setup log rotation
sudo tee /etc/logrotate.d/rese <<EOF
/var/log/rese/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 rese rese
    sharedscripts
    postrotate
        systemctl reload rese > /dev/null 2>&1 || true
    endscript
}
EOF
```

### Systemd Service

Create `/etc/systemd/system/rese.service`:

```ini
[Unit]
Description=RESE API Server
After=network.target

[Service]
Type=notify
User=rese
Group=rese
WorkingDirectory=/opt/rese
Environment="PATH=/opt/rese/venv/bin"
Environment="RESE_CONFIG=/opt/rese/config/production.json"
ExecStart=/opt/rese/venv/bin/uvicorn rese.api:create_app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable rese
sudo systemctl start rese
sudo systemctl status rese
```

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/rese`:

```nginx
upstream rese_backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name rese.example.com;

    client_max_body_size 10M;

    location / {
        proxy_pass http://rese_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }

    location /metrics {
        proxy_pass http://localhost:9090/metrics;
        # Allow only from localhost
        allow 127.0.0.1;
        deny all;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/rese /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rese/ ./rese/
COPY config.json .

# Create non-root user
RUN useradd -m -u 1000 rese && \
    chown -R rese:rese /app
USER rese

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "rese.api:create_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  rese-api:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - RESE_CONFIG=/app/config/production.json
      - RESE_API_KEYS=${RESE_API_KEYS}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config/production.json:/app/config/production.json
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Build and Run

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f rese-api

# Stop services
docker-compose down
```

---

## Cloud Deployment

### AWS Deployment

#### Using ECS (Elastic Container Service)

1. **Push Docker Image to ECR**:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag rese:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rese:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rese:latest
```

2. **Create ECS Task Definition** (`ecs-task-definition.json`):

```json
{
  "family": "rese",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "rese-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/rese:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "RESE_CONFIG",
          "value": "/app/config/production.json"
        }
      ],
      "secrets": [
        {
          "name": "RESE_API_KEYS",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:rese/api-keys"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rese",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

3. **Deploy Task**:

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Run task
aws ecs run-task \
  --cluster rese-cluster \
  --task-definition rese \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### GCP Deployment

#### Using Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/rese

# Deploy to Cloud Run
gcloud run deploy rese \
  --image gcr.io/PROJECT_ID/rese \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars RESE_CONFIG=/app/config/production.json \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10
```

### Azure Deployment

#### Using Container Instances

```bash
# Create resource group
az group create --name rese-rg --location eastus

# Create container registry
az acr create --resource-group rese-rg --registry reseacr --sku Basic

# Build and push image
az acr build --registry reseacr --image rese:latest .

# Deploy container instance
az container create \
  --resource-group rese-rg \
  --name rese-api \
  --image reseacr.azurecr.io/rese:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables RESE_CONFIG=/app/config/production.json
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RESE_CONFIG` | Path to configuration file | `./config.json` |
| `RESE_API_KEYS` | Comma-separated API keys | None |
| `RESE_ENVIRONMENT` | Environment (dev/prod) | `development` |
| `RESE_LOG_LEVEL` | Logging level | `INFO` |

### Configuration File Structure

```json
{
  "environment": "production",
  "project_name": "rese",
  "version": "1.0.0",

  "phase1": {
    "sce_max_constraints": 10000,
    "phi15_enabled": true,
    "phi2_enabled": true
  },

  "phase2": {
    "psi1_enabled": true,
    "psi3_target_accuracy": 0.80
  },

  "phase3": {
    "gamma1_enabled": true,
    "gamma2_iterations": 1000,
    "gamma3_confidence_level": 0.95
  },

  "phase4": {
    "delta3_validation_threshold": 0.7,
    "delta3_min_aci_reduction": 0.2
  },

  "pipeline": {
    "enable_caching": true,
    "cache_ttl_seconds": 3600,
    "max_time_seconds": 86400,
    "max_parallel_tasks": 4
  },

  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "enable_auth": true,
    "rate_limit_per_minute": 60
  },

  "monitoring": {
    "log_level": "INFO",
    "log_file": "/var/log/rese/rese.log",
    "enable_metrics": true,
    "metrics_port": 9090
  }
}
```

---

## Monitoring

### Metrics Endpoint

RESE exposes Prometheus metrics on port 9090:

```
http://localhost:9090/metrics
```

### Key Metrics

- `pipeline_*`: Pipeline execution metrics
- `phase_*`: Phase execution metrics
- `cache_*`: Cache performance
- `error_*`: Error counts
- `aci_*`: ACI tracking

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rese'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

### Grafana Dashboards

Import dashboard from `rese/docs/grafana-dashboard.json`

---

## Troubleshooting

### Common Issues

#### 1. Pipeline Fails to Start

**Symptoms**: API returns 500 error

**Solutions**:
- Check logs: `journalctl -u rese -n 50`
- Verify configuration: `python -c "from rese.config import RESEConfig; print(RESEConfig.load())"`
- Check dependencies: `pip list | grep rese`

#### 2. High Memory Usage

**Solutions**:
- Reduce pipeline workers: `"workers": 2`
- Enable caching: `"enable_caching": true`
- Increase system memory

#### 3. Slow Pipeline Execution

**Solutions**:
- Enable caching: `"use_cache": true`
- Reduce MCTS iterations: `"gamma2_iterations": 500`
- Use GPU if available: `"feature_use_gpu": true`

### Log Analysis

```bash
# View recent logs
sudo journalctl -u rese -n 100 -f

# Search for errors
sudo journalctl -u rese | grep -i error

# View API logs
tail -f /var/log/rese/rese.log
```

### Performance Tuning

```python
# config/tuned.json
{
  "pipeline": {
    "enable_caching": true,
    "max_parallel_tasks": 8,
    "checkpoint_interval": 60
  },
  "phase3": {
    "gamma2_parallel_agents": 8,
    "gamma2_iterations": 500
  }
}
```

---

## Backup and Recovery

### Backup Data

```bash
# Backup configuration and data
tar -czf rese-backup-$(date +%Y%m%d).tar.gz \
  /opt/rese/config \
  /opt/rese/data \
  /var/log/rese

# Copy to backup location
scp rese-backup-*.tar.gz backup-server:/backups/
```

### Restore Data

```bash
# Stop service
sudo systemctl stop rese

# Extract backup
tar -xzf rese-backup-20251231.tar.gz -C /

# Start service
sudo systemctl start rese
```

---

## Security Considerations

1. **API Keys**: Use strong, random API keys
2. **HTTPS**: Enable TLS in production
3. **Firewall**: Restrict access to necessary ports only
4. **Updates**: Keep dependencies updated
5. **Logging**: Monitor logs for suspicious activity

---

## Support

For deployment issues:

- **Documentation**: `rese/docs/`
- **Issues**: GitHub Issues
- **Email**: support@example.com

---

*Last Updated: 2025-12-31*
*Version: 1.0.0*

# Infrastructure Directory

## Purpose

Contains all infrastructure-as-code for deploying and running the OpenEvolve Federation.

## Contents

### Docker Compose

- `docker-compose.yml`: Local development environment
- `docker-compose.prod.yml`: Production configuration
- Service definitions for all adapters and core projects

### Kubernetes (K8s)

- `k8s/`: Kubernetes manifests
- `helm/`: Helm charts for deployment
- ConfigMaps and Secrets

### Networking

- Service discovery configuration
- Network policies
- Ingress rules

### Configuration

- Environment variable templates
- `.env.example`: Reference for required configuration
- Secrets management (SOPS, Vault, etc.)

## Infrastructure Principles

1. **Service Names**: Use Docker service names (e.g., `http://crm-core:8000`)
2. **Dynamic Ports**: Assign ports via environment variables (no hardcoding)
3. **Timeouts**: All HTTP requests must have explicit timeouts (no infinite hangs)
4. **Health Checks**: Every service must expose `/health` endpoint
5. **Circuit Breakers**: Infrastructure-level protection against cascade failures

## Configuration Explicitness

**Every configurable value must be injected via Environment Variables:**

- Ports
- URLs
- Timeouts
- Retries
- API keys
- Database connection strings

**Fail-Safe:** Services must validate `process.env` at startup. If `TARGET_API_URL` is missing, the service MUST crash immediately with a loud error. Do not default to `localhost`.

## Deployment Stages

1. **Development**: `docker-compose up`
2. **Staging**: Kubernetes cluster with replica sets
3. **Production**: High-availability setup with autoscaling

## Service Registration

Each service must:
1. Register with service discovery
2. Expose health check endpoint
3. Report metrics (Prometheus format)
4. Log to centralized collector (ELK, Loki, etc.)

## Security

- Network isolation between adapters
- Auth sidecar for external traffic
- Secrets encryption at rest
- TLS for all inter-service communication

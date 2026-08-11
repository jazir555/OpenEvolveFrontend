# CI/CD Pipeline Documentation

**OpenEvolve Frontend - Continuous Integration/Continuous Deployment**

**Version:** 1.0.0
**Last Updated:** 2025-02-22
**Architecture:** Federation of 30+ Open Source Systems
**Operating Mode:** ZERO TRUST

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Platform-Specific Configurations](#platform-specific-configurations)
5. [Environment Variables](#environment-variables)
6. [Deployment Procedures](#deployment-procedures)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Overview

The OpenEvolve Frontend CI/CD pipeline orchestrates the build, test, and deployment of a hybrid PES-Evolution System that integrates 30+ open source projects. The pipeline follows ZERO TRUST principles, validating everything before deployment.

### Key Principles

- **Runtime Truth:** Trust execution, not documentation
- **Idempotency:** All operations safe to run multiple times
- **Explicit Configuration:** No magic defaults, all config via environment variables
- **Air Gap Enforcement:** Core projects remain immutable
- **Circuit Breakers:** Fail fast, fail gracefully

### Supported Platforms

1. **GitLab CI** - `.gitlab-ci.yml`
2. **GitHub Actions** - `.github/workflows/ci-cd.yml`
3. **Jenkins** - `Jenkinsfile`

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CI/CD Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │ Install  │──▶│Validate  │──▶│  Test    │──▶│ Build  │ │
│  │          │   │          │   │          │   │        │ │
│  └──────────┘   └──────────┘   └──────────┘   └────────┘ │
│                                                  │         │
│                                                  ▼         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │Security  │──▤│Deploy    │──▶│Smoke Test│──▶│Monitor │ │
│  │Scan      │   │          │   │          │   │        │ │
│  └──────────┘   └──────────┘   └──────────┘   └────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Install Dependencies

**Purpose:** Install and cache all Node.js dependencies

**Process:**
1. Checkout repository
2. Setup Node.js (v20)
3. Run `npm ci` for reproducible installs
4. Cache `node_modules` for subsequent stages

**Artifacts:**
- `node_modules/`
- `glue/orchestration/node_modules/`

**Duration:** ~2-3 minutes

---

### Stage 2: Validate

**Purpose:** Ensure code quality and correctness

**Checks:**
- **ESLint:** Code linting with TypeScript rules
- **TypeScript:** Type checking without emit
- **Prettier:** Code formatting validation

**Failure Impact:** Blocks pipeline

---

### Stage 3: Test

#### Unit Tests

**Command:** `npm run test:ci`

**Coverage:**
- Lines: >80%
- Functions: >80%
- Branches: >75%
- Statements: >80%

**Reports:**
- Cobertura XML (for CI integration)
- HTML coverage report

#### Integration Tests

**Command:** `npm run test:e2e`

**Services:**
- Valkey (Redis-compatible) for event bus

**Timeout:** 30 minutes

**Artifacts:**
- Test results JSON
- Screenshots (if applicable)

#### Contract Tests

**Purpose:** Verify API contracts with core projects

**Philosophy:** Runtime Truth - validate actual API behavior

---

### Stage 4: Build

**Purpose:** Compile TypeScript and build Docker images

**Steps:**
1. **TypeScript Build:** Compile all `.ts` files to `.js`
2. **Docker Build:** Create production Docker image
3. **Image Tagging:** Tag with SHA, branch, and latest

**Artifacts:**
- `glue/orchestration/workflows/dist/`
- `glue/adapters/*/dist/`
- Docker image pushed to registry

---

### Stage 5: Security Scan

**Scans:**

1. **Dependency Audit**
   - `npm audit` for vulnerabilities
   - Fail on critical/high severity

2. **Static Analysis (SAST)**
   - Semgrep for security patterns
   - Custom rules for OpenEvolve

3. **License Compliance**
   - Scan for GPL/AGPL licenses
   - Enforce Apache 2.0/MIT/BSD only

---

### Stage 6: Deploy

#### Staging Deployment

**Trigger:** Automatic on `develop` branch

**Environment:**
- Namespace: `openevolve-staging`
- URL: `https://staging.openevolve.io`

**Process:**
1. Configure kubectl
2. Update deployment image
3. Wait for rollout (timeout: 5m)
4. Run smoke tests

#### Production Deployment

**Trigger:** Manual approval on `main` branch

**Environment:**
- Namespace: `openevolve-production`
- URL: `https://openevolve.io`

**Process:**
1. Manual approval gate
2. Configure kubectl
3. Update deployment image
4. Wait for rollout (timeout: 10m)
5. Run smoke tests
6. Monitor for 15 minutes

---

## Platform-Specific Configurations

### GitLab CI

**File:** `.gitlab-ci.yml`

**Features:**
- Docker-in-Docker (DinD) support
- Automatic artifact caching
- Integrated security scanning
- Multi-environment deployments

**Runners:**
- `docker` - For build/test stages
- `kubernetes` - For deployment stages

**Secrets Required:**
- `CI_REGISTRY_USER`
- `CI_REGISTRY_PASSWORD`
- `KUBE_CONTEXT_STAGING`
- `KUBE_CONTEXT_PRODUCTION`

**Usage:**
```bash
# Trigger pipeline
git push origin develop

# Manual deployment
git push origin main
# Then click "Play" on deploy:production job
```

---

### GitHub Actions

**File:** `.github/workflows/ci-cd.yml`

**Features:**
- Caching via `actions/cache`
- Matrix builds (future)
- OIDC for Docker registry auth
- Built-in secret management

**Secrets Required:**
- `KUBE_CONFIG_STAGING` - Base64 encoded kubeconfig
- `KUBE_CONFIG_PRODUCTION` - Base64 encoded kubeconfig
- `GITHUB_TOKEN` - Automatic (for registry auth)

**Usage:**
```bash
# Trigger pipeline
git push origin main

# Manual workflow dispatch
gh workflow run ci-cd.yml
```

---

### Jenkins

**File:** `Jenkinsfile`

**Features:**
- Declarative pipeline syntax
- Parallel stage execution
- Blue Ocean UI compatible
- Plugin-based integrations

**Requirements:**
- Jenkins installed with plugins:
  - `nodejs`
  - `docker-workflow`
  - `kubernetes-cli`
  - `htmlpublisher`

**Credentials Required:**
- `docker-registry` - Docker registry URL
- `docker-credentials` - Username:Password
- `kubeconfig-staging` - Staging kubeconfig
- `kubeconfig-production` - Production kubeconfig

**Usage:**
```bash
# Trigger via webhook
git push origin main

# Manual trigger in Jenkins UI
# Build with Parameters → Select environment
```

---

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `NODE_VERSION` | Node.js version | `20` |
| `KUBECONFIG` | Path to kubeconfig | `/tmp/kubeconfig` |
| `DOCKER_REGISTRY` | Container registry | `ghcr.io` |
| `IMAGE_TAG` | Docker image tag | `v1.2.3` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `HUSKY` | Enable git hooks | `1` |
| `HUSKY_DEBUG` | Debug mode | `0` |
| `npm_config_cache` | npm cache path | `.npm/` |
| `VALKEY_HOST` | Valkey host | `localhost` |
| `VALKEY_PORT` | Valkey port | `6379` |

### Secrets Management

**GitLab:**
```bash
# Via GitLab UI
Settings → CI/CD → Variables → Add Variable

# Protected variables (only main/develop)
# Masked variables (hidden in logs)
```

**GitHub:**
```bash
# Via GitHub CLI
gh secret set KUBE_CONFIG_STAGING < staging-kubeconfig.yaml

# Via GitHub UI
Settings → Secrets and variables → Actions → New repository secret
```

**Jenkins:**
```bash
# Via Jenkins CLI
jenkins-cli create-credentials-by-xml

# Via Jenkins UI
Credentials → Global credentials → Add Credentials
```

---

## Deployment Procedures

### Automated Deployment

**To Staging:**
```bash
# Merge to develop branch
git checkout develop
git merge feature/my-feature
git push origin develop

# Pipeline auto-deploys to staging
```

**To Production:**
```bash
# Merge to main branch
git checkout main
git merge develop
git push origin main

# Manual approval required in CI/CD platform
# Then auto-deploys to production
```

### Manual Deployment

**Using Deployment Script:**
```bash
# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production with specific image
./scripts/deploy.sh production --image-tag=v1.2.3

# Deploy without smoke tests
./scripts/deploy.sh staging --skip-tests

# Check deployment status
./scripts/deploy.sh production --status-only
```

**Using kubectl:**
```bash
# Set context
kubectl config use-context staging

# Update image
kubectl set image deployment/openevolve-frontend \
  openevolve=ghcr.io/openevolve/frontend:v1.2.3 \
  -n openevolve-staging

# Watch rollout
kubectl rollout status deployment/openevolve-frontend \
  -n openevolve-staging

# Check pods
kubectl get pods -n openevolve-staging -l app=openevolve-frontend
```

### Rollback Procedures

**Automatic Rollback:**
The deployment script automatically rolls back if:
- Smoke tests fail (production only)
- Rollout times out
- Health checks fail

**Manual Rollback:**
```bash
# Get previous revision
kubectl rollout history deployment/openevolve-frontend \
  -n openevolve-production

# Rollback to previous revision
kubectl rollout undo deployment/openevolve-frontend \
  -n openevolve-production

# Rollback to specific revision
kubectl rollout undo deployment/openevolve-frontend \
  -n openevolve-production \
  --to-revision=42
```

---

## Troubleshooting

### Common Issues

#### 1. Pipeline Fails at Install Stage

**Symptoms:**
```
npm ci failed
ERESOLVE unable to resolve dependency tree
```

**Solutions:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules
rm -rf node_modules package-lock.json

# Reinstall
npm install

# Update lockfile
npm shadownpm install --package-lock-only
```

---

#### 2. TypeScript Build Fails

**Symptoms:**
```
error TS2307: Cannot find module '@types/node'
```

**Solutions:**
```bash
# Install missing types
npm install --save-dev @types/node

# Rebuild
npm run clean
npm run build
```

---

#### 3. Tests Fail in CI but Pass Locally

**Symptoms:**
Tests pass on machine, fail in pipeline

**Solutions:**
```bash
# Check test environment
npm run test:ci

# Check for timezone issues (always use UTC)
export TZ=UTC

# Check for platform-specific code
npm run test -- --testPathPattern=integration
```

---

#### 4. Docker Build Fails

**Symptoms:**
```
ERROR [build] failed to compute cache key
```

**Solutions:**
```bash
# Check Dockerfile syntax
docker build --check .

# Build without cache
docker build --no-cache -t test .

# Check base image availability
docker pull node:20
```

---

#### 5. Deployment Times Out

**Symptoms:**
```
timed out waiting for the condition
```

**Solutions:**
```bash
# Check pod status
kubectl get pods -n openevolve-staging

# Describe pod
kubectl describe pod/openevolve-frontend-xxx-yyy \
  -n openevolve-staging

# Check logs
kubectl logs pod/openevolve-frontend-xxx-yyy \
  -n openevolve-staging --tail=100

# Check resource limits
kubectl top pods -n openevolve-staging
```

---

#### 6. Smoke Tests Fail After Deployment

**Symptoms:**
Deployment succeeds, smoke tests fail

**Solutions:**
```bash
# Check service endpoints
kubectl get svc -n openevolve-staging

# Port-forward to local
kubectl port-forward svc/openevolve-frontend 8080:80 \
  -n openevolve-staging

# Test locally
curl http://localhost:8080/health

# Check ingress
kubectl get ingress -n openevolve-staging
```

---

### Debug Mode

**Enable verbose logging:**
```bash
# GitLab CI
# In .gitlab-ci.yml, add:
variables:
  CI_DEBUG_TRACE: "true"

# GitHub Actions
# In workflow file, add:
- name: Debug info
  run: |
    echo "Runner OS: ${{ runner.os }}"
    echo "Node version: $(node --version)"
    env | sort

# Jenkins
# In pipeline, add:
options {
    debug()
}
```

---

## Best Practices

### 1. Branch Protection

**GitHub:**
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date
- Restrict who can push to main/develop

**GitLab:**
- Protected branches
- Required approval count
- Push restrictions

### 2. Commit Messages

Follow Conventional Commits:
```
feat(adapters): add LoongFlow adapter implementation
fix(workflows): resolve race condition in event handler
docs: update CI/CD documentation
test: add integration tests for event bus
refactor(api): simplify error handling
```

**Enforced via:** `.husky/commit-msg` hook

### 3. Dependency Management

**Regular updates:**
```bash
# Check for updates
npm outdated

# Update packages
npm update

# Audit for vulnerabilities
npm audit fix

# Check for deprecated packages
npm check --update
```

### 4. Security

**Weekly scans:**
- Run `npm audit`
- Review Semgrep reports
- Check for license violations

**Secret rotation:**
- Rotate API keys quarterly
- Update kubeconfigs after personnel changes
- Review access permissions monthly

### 5. Monitoring

**Post-deployment:**
- Monitor error rates for 15 minutes
- Check response times
- Review application logs
- Verify metrics (Prometheus/Grafana)

**Alerting:**
- Error rate > 1%
- Response time P95 > 500ms
- Pod crash loops
- Health check failures

---

## Appendix

### A. Quick Reference

```bash
# Run full CI locally
npm run validate

# Build locally
npm run build

# Deploy to staging
npm run deploy:staging

# Check deployment status
npm run deploy:status

# Run smoke tests
npm run test:smoke -- --env=production
```

### B. Useful Commands

```bash
# Get latest deployment image
kubectl get deployment openevolve-frontend \
  -n openevolve-production \
  -o jsonpath='{.spec.template.spec.containers[0].image}'

# Watch rollout in real-time
kubectl rollout status deployment/openevolve-frontend \
  -n openevolve-staging \
  --watch=true

# Get pod logs since deployment
kubectl logs -l app=openevolve-frontend \
  -n openevolve-staging \
  --since=10m

# Port-forward to local testing
kubectl port-forward svc/openevolve-frontend 8080:80 \
  -n openevolve-staging
```

### C. Support

**Documentation:**
- `CLAUDE.md` - Architecture and principles
- `README.md` - Project overview
- `CI_CD.md` - This document

**Channels:**
- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Slack: `#openevolve-ci-cd`

**Escalation:**
1. Check pipeline logs
2. Review this documentation
3. Search existing issues
4. Create new issue with template

---

**Document Version:** 1.0.0
**Maintained By:** OpenEvolve Distinguished Engineer
**Review Frequency:** Quarterly
**Next Review:** 2025-05-22

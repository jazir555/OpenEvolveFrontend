# CI/CD Quick Start Guide

**OpenEvolve Frontend - CI/CD Pipeline**

## Quick Start

### For Developers

1. **Install Husky (Git Hooks)**
   ```bash
   npm install
   npx husky install
   ```

2. **Pre-commit Checks**
   - Linting: Auto-runs before commit
   - Type Check: Auto-runs before commit
   - Tests: Auto-runs before commit

3. **Skip Hooks (if needed)**
   ```bash
   git commit --no-verify -m "WIP: my message"
   ```

### For DevOps Engineers

#### GitLab CI

1. **Push to trigger pipeline**
   ```bash
   git push origin develop  # Auto-deploys to staging
   git push origin main     # Manual approval for production
   ```

2. **Manual deployment**
   - Go to: CI/CD → Pipelines
   - Find your pipeline
   - Click "Play" on `deploy:production` job

#### GitHub Actions

1. **Push to trigger workflow**
   ```bash
   git push origin develop  # Auto-deploys to staging
   git push origin main     # Manual approval for production
   ```

2. **Manual deployment**
   ```bash
   gh workflow run ci-cd.yml
   # Or via GitHub UI: Actions → CI/CD Pipeline → Run workflow
   ```

#### Jenkins

1. **Push to trigger build**
   ```bash
   git push origin main
   ```

2. **Manual deployment**
   - Open Jenkins UI
   - Select job: "openevolve-frontend"
   - Click "Build with Parameters"
   - Select environment
   - Click "Build"

### Local Testing

#### Run Tests Locally

```bash
# Full validation
npm run validate

# Individual commands
npm run lint
npm run typecheck
npm run test
npm run build
```

#### Run Smoke Tests

```bash
# Test staging
./scripts/smoke-test.sh -e staging

# Test production
./scripts/smoke-test.sh -e production

# Test with Kubernetes checks
./scripts/smoke-test.sh -e production -k

# Test custom URL
./scripts/smoke-test.sh -u http://localhost:8080

# Verbose mode
./scripts/smoke-test.sh -e staging -v
```

#### Deploy Locally

```bash
# Deploy to staging
npm run deploy:staging

# Deploy to production
npm run deploy:production

# Deploy with specific image
./scripts/deploy.sh production --image-tag=v1.2.3

# Check deployment status
npm run deploy:status
```

## Environment Setup

### Required Secrets

**GitLab:**
- `CI_REGISTRY_USER` - Docker registry username
- `CI_REGISTRY_PASSWORD` - Docker registry password
- `KUBE_CONTEXT_STAGING` - Staging kubeconfig context
- `KUBE_CONTEXT_PRODUCTION` - Production kubeconfig context

**GitHub:**
- `KUBE_CONFIG_STAGING` - Base64 encoded staging kubeconfig
- `KUBE_CONFIG_PRODUCTION` - Base64 encoded production kubeconfig
- `GITHUB_TOKEN` - Automatic (for container registry)

**Jenkins:**
- `docker-registry` - Docker registry URL
- `docker-credentials` - Username:Password
- `kubeconfig-staging` - Staging kubeconfig
- `kubeconfig-production` - Production kubeconfig

### Local Environment

```bash
# Copy example environment file
cp .env.example .env.local

# Edit with your values
nano .env.local
```

## Pipeline Stages

1. **Install** - Install dependencies (~2-3 min)
2. **Validate** - Lint, type check, format check (~1 min)
3. **Test** - Unit, integration, contract tests (~5-10 min)
4. **Build** - Compile TypeScript, build Docker image (~3-5 min)
5. **Security** - Dependency audit, SAST scan (~2-3 min)
6. **Deploy** - Deploy to environment (~2-5 min)

**Total Time:** ~15-30 minutes

## Troubleshooting

### Pipeline Failed?

1. **Check logs**
   - GitLab: CI/CD → Jobs → Click job
   - GitHub: Actions → Click workflow run → Click job
   - Jenkins: Build → Console Output

2. **Common fixes**
   ```bash
   # Dependency issues
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install

   # Build issues
   npm run clean
   npm run build

   # Test issues
   npm run test:verbose
   ```

3. **Get help**
   - Read: `CI_CD.md`
   - Check: GitHub Issues
   - Ask: `#openevolve-ci-cd`

### Rollback

```bash
# Automatic rollback (if smoke tests fail)
./scripts/deploy.sh production  # Auto-rolls back on failure

# Manual rollback
kubectl rollout undo deployment/openevolve-frontend -n openevolve-production

# Check history
kubectl rollout history deployment/openevolve-frontend -n openevolve-production
```

## Best Practices

1. **Branch Protection**
   - Enable on `main` and `develop`
   - Require PR reviews
   - Require status checks

2. **Commit Messages**
   ```
   feat(adapters): add LoongFlow adapter
   fix(workflows): resolve race condition
   docs: update CI/CD docs
   ```

3. **Testing**
   - Write tests for new features
   - Run tests locally before pushing
   - Keep test coverage >80%

4. **Security**
   - Rotate secrets quarterly
   - Review scan reports weekly
   - Keep dependencies updated

## Next Steps

- Read full documentation: `CI_CD.md`
- Read architecture: `CLAUDE.md`
- Check project README: `README.md`

## Support

**Documentation:**
- `CI_CD.md` - Comprehensive CI/CD guide
- `CLAUDE.md` - Architecture principles
- `README.md` - Project overview

**Channels:**
- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Slack: `#openevolve-ci-cd`

---

**Version:** 1.0.0
**Last Updated:** 2025-02-22

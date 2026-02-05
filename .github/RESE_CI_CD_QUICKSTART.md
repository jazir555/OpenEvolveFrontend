# RESE CI/CD Quick Start Guide

This guide will help you get started with the RESE framework CI/CD pipelines.

## Prerequisites

1. **GitHub Repository** with RESE framework code
2. **GitHub Actions** enabled (Settings > Actions > General)
3. **Secrets** configured in GitHub (see below)
4. **Kubernetes Cluster** (for deployment workflow)

---

## Setup (5 minutes)

### Step 1: Configure GitHub Secrets

Go to: **Settings** > **Secrets and variables** > **Actions**

Add the following secrets:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM calls | `sk-...` |
| `GITHUB_TOKEN` | Auto-provided by GitHub Actions | N/A |

Optional secrets for deployment:
| Secret Name | Description |
|------------|-------------|
| `KUBE_CONFIG` | Kubernetes kubeconfig (base64) |
| `REGISTRY_PASSWORD` | Container registry password |

### Step 2: Enable Workflows

Workflows are already in `.github/workflows/`:
- `rese-test.yml` - Testing
- `rese-lint.yml` - Code quality
- `rese-deploy.yml` - Deployment
- `rese-scheduled.yml` - Nightly builds

### Step 3: Validate Workflows (Optional)

```bash
python .github/validate-workflows.py
```

Expected output:
```
RESE Framework CI/CD Workflow Validator
Found 4 workflow file(s)
...
All workflows are valid!
```

---

## Usage

### Running Tests Automatically

Tests run automatically on:
- **Push** to `main` or `develop`
- **Pull Request** to `main` or `develop`

**No action needed!**

### Running Tests Manually

1. Go to **Actions** tab
2. Select **RESE Framework - Testing Pipeline**
3. Click **Run workflow**
4. Select branch and click **Run workflow**

### Deploying to Staging

1. Create a tag:
   ```bash
   git tag rese-v1.0.0-rc1
   git push origin rese-v1.0.0-rc1
   ```

2. Watch **Actions** tab for deployment workflow

3. Or trigger manually:
   - Go to **Actions** > **RESE Framework - Deployment Pipeline**
   - Click **Run workflow**
   - Select environment: `staging`
   - Click **Run workflow**

### Deploying to Production

1. Ensure all tests pass
2. Create production tag:
   ```bash
   git tag rese-v1.0.0
   git push origin rese-v1.0.0
   ```

3. Monitor deployment in **Actions** tab

---

## Common Workflows

### I want to test my changes

**Push to branch:**
```bash
git checkout -b feature/my-feature
# Make changes
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

**Create PR** on GitHub → Tests run automatically

### I want to check code quality

**Run linting locally:**
```bash
# Format code
black glue/adapters/rese-*/src/

# Sort imports
isort glue/adapters/rese-*/src/

# Check style
flake8 glue/adapters/rese-*/src/

# Type check
mypy glue/adapters/rese-*/src/
```

**Or rely on CI** - Push and check Actions tab

### I want to review security

**Wait for nightly build** (2 AM UTC) or trigger manually:
- Actions > RESE Framework - Scheduled Nightly Pipeline > Run workflow

**Check security reports** in Artifacts section

### I want to monitor performance

**Nightly benchmarks** include:
- Phase I throughput (patterns/sec)
- Phase II throughput (domains/sec)
- Phase III throughput (iterations/sec)
- Phase IV throughput (components/sec)

**Compare trends** across runs

---

## Workflow Status

### Check Status in README

Add badges to README.md:
```markdown
![RESE Tests](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/rese-test.yml/badge.svg)
![RESE Lint](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/rese-lint.yml/badge.svg)
```

### Check Status in Terminal

```bash
# List recent workflow runs
gh run list --workflow=rese-test.yml

# Get status of latest run
gh run view --workflow=rese-test.yml

# Watch logs in real-time
gh run watch
```

---

## Troubleshooting

### Test fails but works locally

**Check:**
1. Python version (must be 3.9)
2. Environment variables
3. Dependencies (pip install -r requirements.txt)

**Solution:**
```bash
# Test in CI environment locally
docker run -it python:3.9 bash
# Install deps and run tests
```

### Deployment fails

**Check:**
1. Kubernetes configuration
2. Image registry permissions
3. Resource quotas

**Solution:**
```bash
# Validate manifests locally
kubectl apply --dry-run=client -f infra/k8s-rese-deployment.yaml

# Check pod status
kubectl get pods -n rese-system

# Check logs
kubectl logs -l app=rese-pipeline -n rese-system --tail=100
```

### Workflow doesn't trigger

**Check:**
1. Workflow file syntax (run `validate-workflows.py`)
2. Branch protection rules
3. GitHub Actions permissions

**Solution:**
- Go to Settings > Actions > General
- Enable "Allow all actions and reusable workflows"

### Security scan finds issues

**Common findings:**
- Hardcoded passwords (false positive)
- Use of assert in production
- Weak cryptography

**Solutions:**
- Move secrets to GitHub Secrets
- Replace assert with proper error handling
- Use strong cryptography (bcrypt, etc.)

**False positive?**
- Add `# nosec` comment
- Update Bandit configuration

---

## Best Practices

### For Developers

1. **Write tests** for new features
2. **Run linting** before pushing
3. **Update .env.example** when adding config
4. **Use semantic versioning** for releases

### For DevOps Engineers

1. **Monitor workflows** daily
2. **Review security reports** weekly
3. **Update dependencies** monthly
4. **Tune performance** based on benchmarks

### For Security

1. **Review nightly security reports**
2. **Update dependencies** promptly
3. **Rotate secrets** regularly
4. **Never commit secrets** to repo

---

## Next Steps

1. **Verify setup:** Run a manual workflow
2. **Configure monitoring:** Integrate with Slack/email notifications
3. **Review first nightly build:** Check all reports
4. **Set up branch protection:** Require tests to pass before merge

---

## Getting Help

### Documentation
- Full documentation: `.github/RESE_CI_CD_DOCUMENTATION.md`
- CLAUDE.md principles: `CLAUDE.md`

### GitHub Actions Docs
- https://docs.github.com/en/actions

### Troubleshooting
1. Check **Actions** tab for detailed logs
2. Review workflow artifacts
3. Check Kubernetes logs (if deployment issue)

### Common Commands

```bash
# Validate workflows
python .github/validate-workflows.py

# Trigger test workflow
gh workflow run rese-test.yml

# View workflow runs
gh run list --workflow=rese-test.yml

# Download artifacts
gh run download [run-id]

# Cancel running workflow
gh run cancel [run-id]

# Re-run failed workflow
gh run rerun [run-id]
```

---

## Quick Reference

| Want to... | How |
|-----------|-----|
| Run tests | Push to branch / create PR |
| Check linting | Push and check Actions |
| Deploy to staging | Tag as `rese-v*-rc*` |
| Deploy to production | Tag as `rese-v*` |
| Run nightly build | Actions > Scheduled > Run workflow |
| View reports | Actions > Run > Artifacts |
| Check logs | Actions > Run > Job > Step |
| Cancel run | `gh run cancel [run-id]` |
| Re-run failed | `gh run rerun [run-id]` |

---

## Success Checklist

- [x] Workflows created and validated
- [x] GitHub Secrets configured
- [x] First test run successful
- [x] First deployment successful
- [x] Nightly builds scheduled
- [x] Monitoring configured
- [x] Team trained on workflows

---

**Last Updated:** 2026-02-04
**Version:** 1.0.0

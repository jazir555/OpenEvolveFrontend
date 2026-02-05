# RESE Framework CI/CD Pipelines Documentation

This document describes the comprehensive CI/CD pipelines for the RESE (Research, Engineering, Synthesis, Evaluation) framework.

## Overview

The RESE framework has four automated GitHub Actions workflows:

1. **Testing Pipeline** (`rese-test.yml`) - Main testing on every push/PR
2. **Code Quality Pipeline** (`rese-lint.yml`) - Linting and security checks
3. **Deployment Pipeline** (`rese-deploy.yml`) - Build and deploy to Kubernetes
4. **Scheduled Nightly Pipeline** (`rese-scheduled.yml`) - Full test suite, benchmarks, and security scans

---

## 1. Testing Pipeline (`rese-test.yml`)

**Purpose:** Validate all RESE framework functionality on code changes.

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs:**

### Job 1: Configuration Validation
- Validates all environment variables using `config_validator.py`
- Ensures configuration follows the "Law of Configuration Explicitness"
- Validates all 4 phases + LLTL configuration

### Job 2: Probe Tests
- Runs all phase probes (Phase I-IV)
- Executes full pipeline probe
- Validates each phase independently
- Uploads probe results as artifacts

### Job 3: End-to-End Tests
- Runs complete pipeline integration test
- Tests data flow between phases
- Validates correlation IDs
- Generates comprehensive test report

### Job 4: Integration Tests
- Runs pytest test suite
- Generates coverage reports
- Tests all adapters together

### Job 5: Test Report Aggregation
- Collects all test artifacts
- Generates summary report
- Comments on PRs with test results

### Job 6: Status Check
- Final gatekeeper job
- Ensures all tests passed
- Fails if any job failed

**Artifacts:**
- Configuration validation results (7 days)
- Probe results (7 days)
- E2E test results (30 days)
- Coverage reports (7 days)
- Comprehensive test report (90 days)

---

## 2. Code Quality Pipeline (`rese-lint.yml`)

**Purpose:** Enforce code quality, security standards, and CLAUDE.md compliance.

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs:**

### Job 1: Python Linting
- **Black** - Code formatting check
- **isort** - Import sorting check
- **flake8** - Style guide enforcement (PEP 8)
- **mypy** - Static type checking
- **pylint** - Code quality analysis

### Job 2: Security Scanning
- **Bandit** - Security linter for Python
- **Safety** - Dependency vulnerability scanner
- **pip-audit** - Dependency audit tool

### Job 3: CLAUDE.md Compliance
- Validates "Law of Air Gap" (no imports from core-projects)
- Validates "Law of Configuration Explicitness" (no hardcoded values)
- Validates "Law of Idempotency" (upsert patterns)
- Validates "Law of UTC" (timezone handling)
- Validates structured logging

### Job 4: Environment Variables Validation
- Checks for `.env.example` file
- Validates `config_validator.py` exists
- Checks for hardcoded configuration values

### Job 5: Quality Gate
- Final quality check
- All quality checks must pass
- Fails if any check fails

**Artifacts:**
- Linting reports (7 days)
- Security reports (7 days)
- CLAUDE.md compliance report (30 days)

---

## 3. Deployment Pipeline (`rese-deploy.yml`)

**Purpose:** Build Docker images and deploy to Kubernetes.

**Triggers:**
- Tag push: `rese-v*` (e.g., `rese-v1.0.0`)
- Manual workflow dispatch
  - Can select environment (staging/production)
  - Can skip tests (with warning)

**Jobs:**

### Job 1: Pre-deployment Checks
- Validates configuration
- Verifies Kubernetes manifests
- Dry-run applies all manifests

### Job 2: Build Docker Images
- Builds all phase images (I, II, III, IV)
- Builds adapter images (DEE, LLTL, SCE)
- Uses matrix build for parallel execution
- Pushes to GitHub Container Registry (ghcr.io)
- Caches layers for faster builds

### Job 3: Smoke Tests
- Pulls all built images
- Runs each container with `--help`
- Validates basic functionality
- Uploads smoke test results

### Job 4: Deploy to Kubernetes
- Updates image tags in manifests
- Applies Kubernetes manifests
- Waits for deployments to be ready
- Verifies pod status

### Job 5: Post-deployment Tests
- Runs health checks against all services
- Validates readiness probes
- Monitors pod startup

### Job 6: Deployment Report
- Generates comprehensive deployment report
- Creates GitHub release for tags
- Includes rollback instructions

**Docker Images Built:**
- `ghcr.io/[org]/[repo]/rese-phase1:[tag]`
- `ghcr.io/[org]/[repo]/rese-phase2:[tag]`
- `ghcr.io/[org]/[repo]/rese-phase3:[tag]`
- `ghcr.io/[org]/[repo]/rese-phase4:[tag]`
- `ghcr.io/[org]/[repo]/rese-dee:[tag]`
- `ghcr.io/[org]/[repo]/rese-lltl:[tag]`
- `ghcr.io/[org]/[repo]/rese-sce:[tag]`

**Artifacts:**
- Smoke test results (7 days)
- Deployment manifests (30 days)
- Post-deployment test results (7 days)
- Deployment report (90 days)

---

## 4. Scheduled Nightly Pipeline (`rese-scheduled.yml`)

**Purpose:** Comprehensive testing, benchmarking, and security scanning.

**Triggers:**
- Scheduled: 2 AM UTC every night
- Manual workflow dispatch

**Jobs:**

### Job 1: Full Test Suite
- Runs all unit tests with coverage
- Runs all integration tests
- Runs all probe tests
- Generates comprehensive coverage report
- Runs tests in parallel (pytest-xdist)

### Job 2: Performance Benchmarks
- **Phase I Benchmark** - Varying problem sizes (Small/Medium/Large)
- **Phase II Benchmark** - Varying domain counts
- **Phase III Benchmark** - Varying MCTS iterations
- **Phase IV Benchmark** - Varying component complexity
- Generates trend reports

### Job 3: Security Scanning
- Comprehensive Bandit scan
- Full Safety dependency check
- pip-audit for all dependencies
- TruffleHog secret scanning
- Generates security summary

### Job 4: Dependency Updates Check
- Checks for outdated dependencies
- Generates update recommendations
- Lists breaking changes

### Job 5: Generate Nightly Summary
- Aggregates all results
- Creates comprehensive summary
- Opens GitHub issue on failure
- Archives all reports

**Artifacts:**
- Nightly test results (30 days)
- Performance benchmarks (90 days)
- Security scan results (90 days)
- Dependency report (30 days)
- Nightly summary (90 days)

---

## Concurrency Groups

All workflows use concurrency groups to prevent duplicate runs:

```yaml
concurrency:
  group: rese-[workflow-name]-${{ github.ref }}
  cancel-in-progress: true  # or false for deployment
```

- **Testing & Linting:** Cancel in progress (only latest run matters)
- **Deployment:** Don't cancel (safety)
- **Nightly:** Don't cancel (only one per day)

---

## Environment Variables

All pipelines require these environment variables (set in GitHub Secrets):

### Required
- `OPENAI_API_KEY` - OpenAI API key for LLM calls
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

### Optional (with defaults)
- `RESE_ENV` - Environment (development/staging/production)
- `RESE_LOG_LEVEL` - Logging level (DEBUG/INFO/WARN/ERROR)
- All phase-specific timeout and configuration variables

**Note:** The `config_validator.py` enforces that all required variables are set.

---

## Artifacts and Retention

| Artifact Type | Retention | Location |
|--------------|-----------|----------|
| Test results | 7-30 days | GitHub Actions Artifacts |
| Coverage reports | 7 days | GitHub Actions Artifacts |
| Linting reports | 7 days | GitHub Actions Artifacts |
| Security reports | 90 days | GitHub Actions Artifacts |
| Performance benchmarks | 90 days | GitHub Actions Artifacts |
| Deployment reports | 90 days | GitHub Actions Artifacts |

---

## Caching Strategy

All workflows use GitHub Actions cache for speed:

```yaml
- uses: actions/setup-python@v5
  with:
    cache: 'pip'  # Cache pip dependencies
```

Docker builds also use layer caching:

```yaml
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

---

## Status Badges

Add these badges to your README:

```markdown
![RESE Tests](https://github.com/[org]/[repo]/actions/workflows/rese-test.yml/badge.svg)
![RESE Lint](https://github.com/[org]/[repo]/actions/workflows/rese-lint.yml/badge.svg)
![RESE Deploy](https://github.com/[org]/[repo]/actions/workflows/rese-deploy.yml/badge.svg)
```

---

## Troubleshooting

### Workflow fails to start
- Check workflow syntax: Run `validate-workflows.py`
- Verify GitHub Actions permissions
- Check branch protection rules

### Tests fail locally but pass in CI
- Check environment variable differences
- Verify Python version (must be 3.9)
- Check for missing dependencies

### Deployment fails
- Verify Kubernetes configuration
- Check image registry permissions
- Validate manifests locally: `kubectl apply --dry-run=client`

### Security scan finds vulnerabilities
- Update dependencies: `pip install --upgrade [package]`
- Review Bandit findings (may be false positives)
- Check Safety database for known issues

---

## Best Practices

### For Developers
1. **Run tests locally** before pushing
2. **Use pre-commit hooks** for linting
3. **Update .env.example** when adding new config
4. **Tag releases** properly: `git tag rese-v1.0.0 && git push --tags`

### For DevOps Engineers
1. **Monitor workflow runs** daily
2. **Review security reports** weekly
3. **Update dependencies** monthly
4. **Tune performance** based on benchmarks

### For Security
1. **Review nightly security reports**
2. **Update dependencies promptly**
3. **Rotate secrets regularly**
4. **Monitor for secrets in code**

---

## Maintenance

### Monthly Tasks
- Review and update dependencies
- Check performance trends
- Audit security findings
- Update this documentation

### Quarterly Tasks
- Review and optimize workflows
- Update tool versions (Black, Flake8, etc.)
- Archive old artifacts
- Review retention policies

---

## Getting Help

- **Workflow Issues:** Check Actions tab in GitHub
- **Test Failures:** Review test artifacts
- **Deployment Issues:** Check Kubernetes logs
- **Security Issues:** Review security scan artifacts

For detailed logs:
1. Go to Actions tab
2. Click on failed workflow run
3. Click on failed job
4. Click on failed step
5. Review logs

---

## Future Enhancements

Potential improvements:
- [ ] Add performance regression detection
- [ ] Integrate with SonarQube
- [ ] Add automated changelog generation
- [ ] Add multi-region deployment support
- [ ] Add canary deployment support
- [ ] Integrate with monitoring (Prometheus/Grafana)
- [ ] Add automated rollback on failure
- [ ] Add load testing to nightly builds

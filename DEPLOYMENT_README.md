# Sovereign System Deployment Guide

## Quick Start

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Deploy to development
python deploy.py --environment development

# Start the system
./start_sovereign.sh  # Unix
start_sovereign.bat   # Windows
```

### Docker Deployment

```bash
# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## Deployment Options

### 1. Manual Deployment

Use the deployment script for full control:

```bash
# Development environment
python deploy.py --environment development

# Staging environment
python deploy.py --environment staging

# Production environment
python deploy.py --environment production

# Skip tests (faster)
python deploy.py --environment production --skip-tests
```

### 2. Docker Deployment

Containerized deployment for consistency:

```bash
# Build image
docker build -t sovereign-system:latest .

# Run container
docker run -d -p 8501:8501 -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  sovereign-system:latest

# Or use docker-compose
docker-compose up -d
```

### 3. Makefile Commands

Simplified commands using Make:

```bash
make install        # Install dependencies
make test           # Run all tests
make deploy-prod    # Deploy to production
make docker-up      # Start Docker containers
make health-check   # Check system health
```

## Configuration

### Environment-Specific Configs

Configuration files for each environment:
- `deploy_config_development.json` - Development settings
- `deploy_config_staging.json` - Staging settings
- `deploy_config_production.json` - Production settings

### Key Configuration Parameters

```json
{
  "python_version": "3.11",
  "database_path": "sovereign_system.db",
  "log_level": "INFO",
  "api_port": 8501,
  "cache_size": 1000,
  "max_concurrent_problems": 100,
  "quality_thresholds": {
    "coherence": 0.85,
    "completeness": 0.90,
    "feasibility": 0.80
  }
}
```

## Deployment Process

### Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Configuration files updated
- [ ] Backup created
- [ ] Rollback plan ready

### Deployment Steps

1. **Prerequisites Check**
   - Python version
   - Required files
   - System resources

2. **Install Dependencies**
   - Install from requirements.txt
   - Verify installations

3. **Initialize Database**
   - Create schema
   - Run migrations

4. **Run Tests** (optional)
   - Unit tests
   - Integration tests
   - Smoke tests

5. **Create Configuration**
   - Environment variables
   - Logging config
   - Startup scripts

6. **Setup Monitoring**
   - Health checks
   - Performance monitoring
   - Error tracking

### Post-Deployment

1. Verify health checks pass
2. Test critical workflows
3. Monitor logs for errors
4. Check performance metrics

## Rollback Procedure

If deployment fails:

```bash
# Automatic rollback
python deploy.py --rollback

# Manual rollback
1. Stop the system
2. Restore database backup
3. Revert code changes
4. Restart system
```

## Monitoring

### Health Checks

```python
from sovereign_reliability import get_health_monitor

monitor = get_health_monitor()
results = monitor.run_health_checks()
print(results)
```

### Performance Monitoring

```python
from sovereign_performance_optimization import get_performance_stats

stats = get_performance_stats()
for operation, metrics in stats.items():
    print(f"{operation}: {metrics['avg_duration']:.2f}s")
```

### Logs

- Application logs: `sovereign_system.log`
- Docker logs: `docker-compose logs -f`
- System logs: Check `logs/` directory

## Troubleshooting

### Common Issues

**Issue**: Deployment fails at prerequisites
**Solution**: Check Python version and required files

**Issue**: Database initialization fails
**Solution**: Check file permissions and disk space

**Issue**: Tests fail during deployment
**Solution**: Review test output, fix issues, redeploy

**Issue**: Health checks fail
**Solution**: Check logs, verify dependencies, restart

### Debug Mode

Enable debug logging:

```bash
# Set in config
"log_level": "DEBUG"

# Or environment variable
export LOG_LEVEL=DEBUG
```

## CI/CD Integration

### GitHub Actions

Automated CI/CD pipeline in `.github/workflows/ci.yml`:

- Runs on push to main/develop
- Tests on multiple OS and Python versions
- Deploys to staging (develop) or production (main)
- Creates releases automatically

### Manual Trigger

```bash
# Push to trigger CI/CD
git push origin main

# Or create a release
git tag v1.0.0
git push origin v1.0.0
```

## Security

### Best Practices

1. Use environment variables for secrets
2. Enable HTTPS in production
3. Regular security updates
4. Monitor for vulnerabilities
5. Implement rate limiting

### Security Scanning

```bash
# Run security scan
bandit -r . -f json -o security-report.json

# Check dependencies
safety check
```

## Scaling

### Horizontal Scaling

Deploy multiple instances behind a load balancer:

```bash
# Start multiple containers
docker-compose up -d --scale sovereign-system=3
```

### Performance Tuning

Adjust configuration for scale:
- Increase `max_concurrent_problems`
- Increase `cache_size`
- Optimize database queries
- Enable connection pooling

## Backup and Recovery

### Automated Backups

```bash
# Create backup
make backup

# Restore from backup
make restore BACKUP=sovereign_system_backup_20241023.db
```

### Backup Schedule

- Daily: Automated database backup
- Weekly: Full system backup
- Monthly: Archive old backups

## Support

For deployment issues:
1. Check this guide
2. Review `OPERATIONS_GUIDE.md`
3. Check system logs
4. Contact development team

## Version History

- v1.0.0 - Initial production release
- All 198 tests passing
- Complete feature set implemented

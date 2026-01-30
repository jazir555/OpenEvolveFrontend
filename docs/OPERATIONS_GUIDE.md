# Sovereign System Operations Guide

## Monitoring and Operations

### Health Monitoring

#### Running Health Checks

```python
from sovereign_reliability import get_health_monitor

monitor = get_health_monitor()
results = monitor.run_health_checks()

if results['overall_healthy']:
    print("System is healthy")
else:
    print(f"System unhealthy: {results['failed_checks']}")
```

#### Registering Custom Health Checks

```python
def check_api_endpoint():
    """Check if API is responding."""
    try:
        import requests
        response = requests.get("http://localhost:8501/health", timeout=5)
        return response.status_code == 200
    except:
        return False

monitor.register_check("api_endpoint", check_api_endpoint)
```

### Performance Monitoring

#### Tracking Operation Performance

```python
from sovereign_performance_optimization import get_performance_stats

stats = get_performance_stats()
for operation, metrics in stats.items():
    print(f"{operation}:")
    print(f"  Count: {metrics['count']}")
    print(f"  Avg Duration: {metrics['avg_duration']:.2f}s")
    print(f"  Total Duration: {metrics['total_duration']:.2f}s")
```

#### Setting Up Alerts

Monitor key metrics and alert when thresholds are exceeded:

```python
def check_performance_degradation():
    stats = get_performance_stats()
    if 'decompose' in stats:
        avg_time = stats['decompose']['avg_duration']
        if avg_time > 30.0:  # Alert if > 30s
            send_alert(f"Decomposition slow: {avg_time:.2f}s")
```


### Error Handling and Recovery

#### Viewing Error Logs

```python
from sovereign_reliability import get_error_handler

handler = get_error_handler()
stats = handler.get_error_stats()

print(f"Total errors: {stats['total_errors']}")
print(f"Error breakdown: {stats['error_counts']}")
print(f"Recent errors: {stats['recent_errors']}")
```

#### Implementing Retry Logic

```python
from sovereign_reliability import with_retry

@with_retry(max_attempts=3, retry_on=(ConnectionError, TimeoutError))
def unreliable_operation():
    # Your code here
    pass
```

### Database Operations

#### Backup Database

```bash
# Create backup
cp sovereign_system.db sovereign_system_backup_$(date +%Y%m%d).db

# Restore from backup
cp sovereign_system_backup_20241023.db sovereign_system.db
```

#### Database Maintenance

```python
from sovereign_persistence import SovereignDatabase

db = SovereignDatabase()

# Vacuum database to reclaim space
db.connection.execute("VACUUM")

# Analyze for query optimization
db.connection.execute("ANALYZE")
```

### Incident Response

#### System Unresponsive

1. Check health status
2. Review error logs
3. Check resource usage (CPU, memory)
4. Restart system if needed

#### High Error Rate

1. Check error handler stats
2. Identify error patterns
3. Review recent changes
4. Apply fixes or rollback

#### Performance Degradation

1. Check performance stats
2. Review cache hit rates
3. Check database query performance
4. Scale resources if needed

### Maintenance Tasks

#### Daily
- Monitor health checks
- Review error logs
- Check performance metrics

#### Weekly
- Backup database
- Review system capacity
- Update dependencies

#### Monthly
- Performance optimization review
- Security audit
- Capacity planning

### Deployment Procedures

#### Standard Deployment

```bash
python deploy.py --environment production
```

#### Rollback Procedure

```bash
python deploy.py --rollback
```

#### Zero-Downtime Deployment

1. Deploy to staging
2. Run smoke tests
3. Switch traffic gradually
4. Monitor for issues
5. Complete or rollback

### Troubleshooting

#### Common Issues

**Issue**: Database locked
**Solution**: Close all connections, restart system

**Issue**: Memory usage high
**Solution**: Clear caches, restart system, scale resources

**Issue**: Slow decomposition
**Solution**: Check cache, optimize queries, increase resources

### Monitoring Dashboards

Access monitoring at:
- Health: `http://localhost:8501/health`
- Metrics: `http://localhost:8501/metrics`
- Logs: `sovereign_system.log`

### Contact and Escalation

For critical issues:
1. Check this guide
2. Review API documentation
3. Check system logs
4. Escalate to development team

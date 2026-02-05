# RESE Framework - Configuration Quick Reference

## Quick Start

```bash
# 1. Copy example configuration
cp .env.example .env

# 2. Edit .env with your values
# 3. Validate configuration
python -m glue.adapters.rese_integration.config_validator

# 4. Use in your code
from glue.adapters.rese_integration.config_loader import load_config
config = load_config()
```

## Most Common Variables

### Required (No Defaults)

```bash
# Environment
RESE_ENV=production
RESE_LOG_LEVEL=WARN

# Phase I
PHASE1_TIMEOUT_MS=60000
PHASE1_MAX_ASSUMPTIONS=500
PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.8

# Phase III
PHASE3_TIMEOUT_MS=300000
PHASE3_ITERATIONS=100000
PHASE3_PARALLEL_WORKERS=8

# External Services
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379/0
```

### Performance Tuning

```bash
# Faster (Development)
PHASE1_TIMEOUT_MS=10000
PHASE3_ITERATIONS=5000
PHASE4_VALIDATION_LEVEL=1

# Slower (Production)
PHASE1_TIMEOUT_MS=60000
PHASE3_ITERATIONS=100000
PHASE4_VALIDATION_LEVEL=3
```

## Validation Rules

At startup, the adapter validates:

1. ✅ All required variables present
2. ✅ Numeric values within ranges
3. ✅ Enum values match allowed options
4. ✅ URLs well-formed
5. ✅ File paths accessible

**Validation fails → Application crashes**

## Environment Profiles

### Development
```bash
RESE_ENV=development
RESE_LOG_LEVEL=DEBUG
PHASE1_TIMEOUT_MS=10000
PHASE3_ITERATIONS=5000
PHASE4_VALIDATION_LEVEL=1
PHASE1_ENABLE_LEAN4_INTEGRATION=false
ENABLE_PROFILING=true
```

### Production
```bash
RESE_ENV=production
RESE_LOG_LEVEL=WARN
PHASE1_TIMEOUT_MS=60000
PHASE3_ITERATIONS=100000
PHASE4_VALIDATION_LEVEL=3
PHASE1_ENABLE_LEAN4_INTEGRATION=true
ENABLE_CIRCUIT_BREAKERS=true
ENABLE_DLQ=true
```

## Key Trade-offs

### Speed vs Quality

| Variable | Fast (Dev) | Slow (Prod) |
|----------|------------|-------------|
| `PHASE1_TIMEOUT_MS` | 10,000 | 60,000 |
| `PHASE3_ITERATIONS` | 5,000 | 100,000 |
| `PHASE4_VALIDATION_LEVEL` | 1 | 3 |

### Recall vs Precision

| Variable | High Recall | High Precision |
|----------|-------------|----------------|
| `PHASE1_MIN_ASSUMPTION_CONFIDENCE` | 0.5 | 0.9 |
| `PHASE2_IMECH_THRESHOLD` | 0.3 | 0.9 |
| `PHASE4_MIN_CONFIDENCE_THRESHOLD` | 0.6 | 0.95 |

### Exploration vs Exploitation (Phase III)

| Variable | Explore | Exploit |
|----------|---------|---------|
| `PHASE3_UCB1_C` | 2.5 | 0.5 |
| `PHASE3_CONVERGENCE_THRESHOLD` | 0.01 | 0.0001 |

## Troubleshooting

### Error: Missing Required Variable

```bash
# Add to .env
OPENAI_API_KEY=sk-your-key-here
```

### Error: Value Out of Range

```bash
# Adjust value
PHASE1_TIMEOUT_MS=30000  # Must be between 1000 and 300000
```

### Phase I Too Slow

```bash
PHASE1_MAX_ASSUMPTIONS=50
PHASE1_ENABLE_TACIT_MINING=false
PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.5
```

### Phase III Too Slow

```bash
PHASE3_ITERATIONS=5000
PHASE3_PARALLEL_WORKERS=2
PHASE3_CONVERGENCE_THRESHOLD=0.01
```

## Security Checklist

- [ ] Never commit `.env` files
- [ ] Use secrets manager in production
- [ ] Rotate API keys regularly
- [ ] Use read-only Redis accounts for GET operations
- [ ] Enable HTTPS for external services
- [ ] Set up authentication for Redis

## Full Documentation

See [README_CONFIG.md](README_CONFIG.md) for complete documentation.

## Support

- Validate: `python -m glue.adapters.rese_integration.config_validator`
- Test: `python -m glue.adapters.rese_integration.config_example`
- Issues: [GitHub Issues](https://github.com/your-org/rese-adapter/issues)

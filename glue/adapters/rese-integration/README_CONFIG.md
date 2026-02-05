# RESE Framework - Configuration Guide

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration Principles](#configuration-principles)
- [Phase I Configuration](#phase-i-epistemic-audit)
- [Phase II Configuration](#phase-ii-isomorphic-mapping)
- [Phase III Configuration](#phase-iii-mcts-search)
- [Phase IV Configuration](#phase-iv-architecture-assembly)
- [Supporting Components](#supporting-components)
- [Environment Profiles](#environment-profiles)
- [Validation and Troubleshooting](#validation-and-troubleshooting)

---

## Overview

This guide documents all environment variables for the RESE (Rational Epistemic Synthesis Engine) integration adapter. Following the **Law of Configuration Explicitness**, there are NO magic defaults - every configurable value must be explicitly set via environment variables.

### Key Principles

1. **Fail Fast**: The adapter crashes at startup if required variables are missing
2. **Explicit Values**: No implicit defaults - you must declare your intent
3. **Validation**: All values are validated for type, range, and format
4. **Documentation**: Every variable has clear comments explaining its purpose

---

## Quick Start

### 1. Copy the Example Configuration

```bash
cp .env.example .env
```

### 2. Edit for Your Environment

```bash
# Required: At minimum, set these variables
RESE_ENV=development
OPENAI_API_KEY=sk-your-key-here
REDIS_URL=redis://localhost:6379/0
```

### 3. Validate Configuration

```bash
# Run the configuration validator
python -m glue.adapters.rese_integration.config_validator
```

### 4. Start the Adapter

```bash
# If validation passes, start the service
python -m glue.adapters.rese_integration.main
```

---

## Configuration Principles

### The "Law of Configuration Explicitness"

**Philosophy**: Implicit defaults are dangerous. They lead to:

- Configuration drift between environments
- Unexpected behavior in production
- Difficult debugging ("why did it use port 8080?")
- Security vulnerabilities (default credentials)

**Our Approach**:

- ✅ Every value must be explicitly set
- ✅ All values are validated at startup
- ✅ Missing required values cause immediate crash
- ✅ Clear error messages identify what's missing

### Variable Naming Convention

All RESE variables follow this pattern:

```
{PHASE}_{NAME}_{UNIT/TYPE}
```

Examples:
- `PHASE1_TIMEOUT_MS` - Phase 1 timeout in milliseconds
- `PHASE3_UCB1_C` - Phase 3 UCB1 constant (dimensionless)
- `LLTL_ENCODING_DIM` - LLTL encoding dimension (count)

### Units

Variables always specify units in the name:

- `_MS` - milliseconds
- `_S` or `_SECONDS` - seconds
- `_COUNT` - integer count
- `_THRESHOLD` - float 0.0 to 1.0
- `_LEVEL` - enum or integer level

---

## Phase I Configuration

### Epistemic Audit

Purpose: Extract and validate assumptions from problem description.

#### PHASE1_TIMEOUT_MS

**Purpose**: Maximum time to spend extracting assumptions

**Range**: 1,000 - 300,000 ms (1 second to 5 minutes)

**Development**: 30,000 ms (30 seconds)
**Production**: 60,000 ms (1 minute)

**Trade-off**: Longer timeouts allow deeper analysis but increase latency

```bash
# Quick development iteration
PHASE1_TIMEOUT_MS=10000

# Comprehensive production analysis
PHASE1_TIMEOUT_MS=120000
```

#### PHASE1_MAX_ASSUMPTIONS

**Purpose**: Limit the number of assumptions extracted to prevent combinatorial explosion

**Range**: 10 - 1,000

**Development**: 100
**Production**: 500

**Trade-off**: More assumptions = more comprehensive but slower downstream processing

```bash
# Small problems (e.g., simple algorithm design)
PHASE1_MAX_ASSUMPTIONS=50

# Large problems (e.g., system architecture)
PHASE1_MAX_ASSUMPTIONS=500
```

#### PHASE1_MIN_ASSUMPTION_CONFIDENCE

**Purpose**: Filter out low-confidence assumptions

**Range**: 0.0 - 1.0

**Development**: 0.6 (60% confidence)
**Production**: 0.8 (80% confidence)

**Impact**: Assumptions below this threshold are flagged for manual review

```bash
# Exploratory mode (include borderline assumptions)
PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.5

# Conservative mode (only high-confidence assumptions)
PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.9
```

#### PHASE1_CIRCUIT_BREAKER_THRESHOLD

**Purpose**: Consecutive failures before opening circuit breaker

**Range**: 1 - 100

**Development**: 5
**Production**: 3

**Behavior**: After this many failures, Phase I is temporarily disabled to prevent cascading failures

```bash
# Lenient (allow more failures before tripping)
PHASE1_CIRCUIT_BREAKER_THRESHOLD=10

# Strict (trip immediately on failures)
PHASE1_CIRCUIT_BREAKER_THRESHOLD=1
```

#### Boolean Flags

```bash
# Enable advanced features (trade computation for quality)
PHASE1_ENABLE_TACIT_MINING=true      # Extract unstated assumptions
PHASE1_ENABLE_RED_TEAM=true           # Find contradicting assumptions
PHASE1_ENABLE_LEAN4_INTEGRATION=false # Formal verification (slow!)
```

**Recommendations**:
- **Development**: Enable all flags except Lean4 (for speed)
- **Staging**: Enable all flags (for comprehensive testing)
- **Production**: Enable all flags (for maximum quality)

---

## Phase II Configuration

### Isomorphic Mapping

Purpose: Map problem domain to target solution domains.

#### PHASE2_TIMEOUT_MS

**Purpose**: Maximum time for isomorphic search

**Range**: 1,000 - 600,000 ms (1 second to 10 minutes)

**Development**: 60,000 ms (1 minute)
**Production**: 120,000 ms (2 minutes)

```bash
# Fast mode (few domains, shallow search)
PHASE2_TIMEOUT_MS=30000

# Comprehensive mode (many domains, deep search)
PHASE2_TIMEOUT_MS=300000
```

#### PHASE2_IMECH_THRESHOLD

**Purpose**: Strictness for isomorphism matching

**Range**: 0.0 - 1.0

**Development**: 0.5 (moderate strictness)
**Production**: 0.7 (high strictness)

**Trade-off**:
- Higher values = fewer, higher-quality mappings
- Lower values = more mappings, but some may be spurious

```bash
# Permissive mode (find many potential mappings)
PHASE2_IMECH_THRESHOLD=0.3

# Strict mode (only strong matches)
PHASE2_IMECH_THRESHOLD=0.9
```

#### PHASE2_MAX_TARGET_DOMAINS

**Purpose**: Limit search space to prevent combinatorial explosion

**Range**: 1 - 50

**Development**: 5
**Production**: 10

**Impact**: Each additional domain multiplies search time exponentially

```bash
# Narrow search (specific domain)
PHASE2_MAX_TARGET_DOMAINS=2

# Broad search (cross-domain innovation)
PHASE2_MAX_TARGET_DOMAINS=20
```

#### PHASE2_SEARCH_DEPTH

**Purpose**: How deep to search for isomorphic structures

**Range**: 1 - 20

**Development**: 3
**Production**: 5

**Trade-off**: Deeper search = exponentially slower but more comprehensive

```bash
# Shallow search (fast, surface-level patterns)
PHASE2_SEARCH_DEPTH=2

# Deep search (slow, fundamental structures)
PHASE2_SEARCH_DEPTH=8
```

---

## Phase III Configuration

### MCTS Search

Purpose: Monte Carlo Tree Search for solution space exploration.

#### PHASE3_TIMEOUT_MS

**Purpose**: Maximum time for MCTS exploration

**Range**: 1,000 - 3,600,000 ms (1 second to 1 hour)

**Development**: 120,000 ms (2 minutes)
**Production**: 300,000 ms (5 minutes)

**Note**: This is typically the longest phase - allocate time accordingly

```bash
# Quick exploration (prototype)
PHASE3_TIMEOUT_MS=60000

# Deep exploration (production)
PHASE3_TIMEOUT_MS=600000
```

#### PHASE3_ITERATIONS

**Purpose**: Soft limit on MCTS iterations (actual count limited by timeout)

**Range**: 100 - 10,000,000

**Development**: 10,000
**Production**: 100,000

**Relationship**: Actual iterations = min(TIMEOUT_LIMIT, ITERATION_LIMIT)

```bash
# Few iterations (fast, coarse-grained search)
PHASE3_ITERATIONS=1000

# Many iterations (slow, fine-grained search)
PHASE3_ITERATIONS=1000000
```

#### PHASE3_UCB1_C

**Purpose**: Exploration vs exploitation balance in MCTS

**Range**: 0.0 - 10.0

**Standard**: 1.414 (√2)

**Behavior**:
- Higher values = more exploration (try new nodes)
- Lower values = more exploitation (refine known good nodes)

```bash
# Explore heavily (find diverse solutions)
PHASE3_UCB1_C=2.5

# Exploit heavily (refine best solutions)
PHASE3_UCB1_C=0.5

# Balanced (theoretical optimum)
PHASE3_UCB1_C=1.414
```

#### PHASE3_PARALLEL_WORKERS

**Purpose**: Number of parallel MCTS simulations

**Range**: 1 - 64

**Development**: 4
**Production**: CPU count

**Performance**: Near-linear speedup until CPU saturation

```bash
# Single-threaded (debugging)
PHASE3_PARALLEL_WORKERS=1

# Multi-threaded (production)
PHASE3_PARALLEL_WORKERS=16
```

---

## Phase IV Configuration

### Architecture Assembly

Purpose: Assemble final solution from selected components.

#### PHASE4_VALIDATION_LEVEL

**Purpose**: How rigorously to validate the final architecture

**Range**: 0 - 3

**Levels**:
- 0 = No validation (fastest, least safe)
- 1 = Syntax/type checking only
- 2 = Semantic validation (logical consistency)
- 3 = Full formal verification (slowest, safest)

**Development**: 2
**Production**: 3

```bash
# Fast mode (skip formal verification)
PHASE4_VALIDATION_LEVEL=1

# Safe mode (full verification)
PHASE4_VALIDATION_LEVEL=3
```

#### PHASE4_INTEGRATION_STRATEGY

**Purpose**: How aggressively to integrate components

**Values**: `conservative`, `balanced`, `aggressive`

**Development**: `balanced`
**Production**: `balanced`

**Behavior**:
- Conservative: Only 100% confidence components
- Balanced: 80%+ confidence components
- Aggressive: All viable components

```bash
# Risk-averse (only high-confidence components)
PHASE4_INTEGRATION_STRATEGY=conservative

# Risk-tolerant (explore novel combinations)
PHASE4_INTEGRATION_STRATEGY=aggressive
```

---

## Supporting Components

### LLTL (Lean Temporal Logic)

#### LLTL_ENCODING_DIM

**Purpose**: Vector dimension for temporal logic embeddings

**Range**: 64 - 4,096

**Development**: 512
**Production**: 1,024

**Trade-off**: Higher dimensions = more expressive but slower

```bash
# Fast encoding
LLTL_ENCODING_DIM=256

# Rich encoding
LLTL_ENCODING_DIM=2048
```

---

## Environment Profiles

### Development Profile

**Goal**: Fast iteration, detailed logging, permissive validation

```bash
# Environment
RESE_ENV=development
RESE_LOG_LEVEL=DEBUG

# Timeouts (short)
PHASE1_TIMEOUT_MS=10000
PHASE2_TIMEOUT_MS=30000
PHASE3_TIMEOUT_MS=60000
PHASE4_TIMEOUT_MS=20000

# Quality thresholds (lower)
PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.5
PHASE2_IMECH_THRESHOLD=0.4
PHASE4_MIN_CONFIDENCE_THRESHOLD=0.6

# Features (enable all except slow ones)
PHASE1_ENABLE_TACIT_MINING=true
PHASE1_ENABLE_RED_TEAM=true
PHASE1_ENABLE_LEAN4_INTEGRATION=false  # Too slow for dev

# Validation (light)
PHASE4_VALIDATION_LEVEL=1

# Observability (detailed)
ENABLE_PROFILING=true
ENABLE_METRICS=true
ENABLE_TRACING=false
```

### Staging Profile

**Goal**: Production-like configuration with comprehensive testing

```bash
# Environment
RESE_ENV=staging
RESE_LOG_LEVEL=INFO

# Timeouts (moderate)
PHASE1_TIMEOUT_MS=30000
PHASE2_TIMEOUT_MS=60000
PHASE3_TIMEOUT_MS=120000
PHASE4_TIMEOUT_MS=30000

# Quality thresholds (moderate)
PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.7
PHASE2_IMECH_THRESHOLD=0.6
PHASE4_MIN_CONFIDENCE_THRESHOLD=0.75

# Features (enable all)
PHASE1_ENABLE_TACIT_MINING=true
PHASE1_ENABLE_RED_TEAM=true
PHASE1_ENABLE_LEAN4_INTEGRATION=true

# Validation (full)
PHASE4_VALIDATION_LEVEL=3

# Observability (full)
ENABLE_PROFILING=true
ENABLE_METRICS=true
ENABLE_TRACING=true
```

### Production Profile

**Goal**: Maximum quality, resilience, and observability

```bash
# Environment
RESE_ENV=production
RESE_LOG_LEVEL=WARN

# Timeouts (longer for quality)
PHASE1_TIMEOUT_MS=60000
PHASE2_TIMEOUT_MS=120000
PHASE3_TIMEOUT_MS=300000
PHASE4_TIMEOUT_MS=60000

# Quality thresholds (high)
PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.8
PHASE2_IMECH_THRESHOLD=0.7
PHASE4_MIN_CONFIDENCE_THRESHOLD=0.85

# Features (enable all)
PHASE1_ENABLE_TACIT_MINING=true
PHASE1_ENABLE_RED_TEAM=true
PHASE1_ENABLE_LEAN4_INTEGRATION=true

# Validation (full)
PHASE4_VALIDATION_LEVEL=3
PHASE4_INTEGRATION_STRATEGY=conservative

# Resilience
ENABLE_CIRCUIT_BREAKERS=true
CIRCUIT_BREAKER_RESET_TIMEOUT_MS=60000
ENABLE_RETRY=true
MAX_RETRY_ATTEMPTS=3
ENABLE_DLQ=true

# Observability (production-grade)
ENABLE_PROFILING=false
ENABLE_METRICS=true
ENABLE_TRACING=true
```

---

## Validation and Troubleshooting

### Configuration Validation

The adapter validates configuration at startup. It checks:

1. **Presence**: All required variables are set
2. **Type**: Numbers are numeric, booleans are true/false
3. **Range**: Values are within min/max bounds
4. **Format**: URLs are well-formed, UUIDs are valid
5. **Access**: File paths exist and are readable

### Running the Validator

```bash
# Validate configuration without starting the service
python -m glue.adapters.rese_integration.config_validator

# Validate specific profile
RESE_ENV=production python -m glue.adapters.rese_integration.config_validator
```

### Common Validation Errors

#### Error: Missing Required Variable

```
CRITICAL: Configuration validation failed
Error: Missing required variable: OPENAI_API_KEY
Fix: Add OPENAI_API_KEY to your .env file
```

**Solution**: Add the missing variable to `.env`

#### Error: Value Out of Range

```
CRITICAL: Configuration validation failed
Error: PHASE1_TIMEOUT_MS=500000 exceeds maximum of 300000
Fix: Use a value between 1000 and 300000
```

**Solution**: Adjust the value to be within the valid range

#### Error: Invalid Enum Value

```
CRITICAL: Configuration validation failed
Error: RESE_ENV='prod' is not one of [development, staging, production]
Fix: Use a valid value: development, staging, or production
```

**Solution**: Use one of the allowed enum values

### Debugging Configuration

#### 1. Check Current Configuration

```bash
# Print all loaded configuration (with masking for secrets)
python -m glue.adapters.rese_integration.config_dump

# Print specific section
python -m glue.adapters.rese_integration.config_dump --section phase1
```

#### 2. Test Connectivity

```bash
# Test connection to external services
python -m glue.adapters.rese_integration.test_connections
```

#### 3. Dry Run

```bash
# Run configuration validation without starting service
RESE_ENV=production python -m glue.adapters.rese_integration.dry_run
```

### Performance Tuning

#### If Phase I is Too Slow

```bash
# Reduce assumption count
PHASE1_MAX_ASSUMPTIONS=50

# Disable tacit mining
PHASE1_ENABLE_TACIT_MINING=false

# Lower confidence threshold
PHASE1_MIN_ASSUMPTION_CONFIDENCE=0.5
```

#### If Phase II is Too Slow

```bash
# Reduce target domains
PHASE2_MAX_TARGET_DOMAINS=3

# Shallow search
PHASE2_SEARCH_DEPTH=2

# Lower pattern threshold
PHASE2_PATTERN_THRESHOLD=0.5
```

#### If Phase III is Too Slow

```bash
# Reduce iterations
PHASE3_ITERATIONS=5000

# Fewer workers
PHASE3_PARALLEL_WORKERS=2

# Lower convergence threshold
PHASE3_CONVERGENCE_THRESHOLD=0.01
```

### Security Best Practices

1. **Never Commit `.env` Files**
   ```bash
   # Add .env to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use Secrets Management in Production**
   ```bash
   # AWS Secrets Manager
   # HashiCorp Vault
   # Azure Key Vault
   ```

3. **Rotate Keys Regularly**
   ```bash
   # Set up automated key rotation for:
   # - OPENAI_API_KEY
   # - REDIS_URL (if authenticated)
   ```

4. **Use Read-Only Accounts Where Possible**
   ```bash
   # Redis should use read-only credentials for GET operations
   # Only write operations need full credentials
   ```

---

## Summary

### Configuration Checklist

- [ ] Copy `.env.example` to `.env`
- [ ] Set `RESE_ENV` (development/staging/production)
- [ ] Configure `OPENAI_API_KEY` or `LOCAL_LLM_ENDPOINT`
- [ ] Set `REDIS_URL` for caching
- [ ] Adjust timeouts based on SLA requirements
- [ ] Configure quality thresholds based on risk tolerance
- [ ] Enable/disable features based on performance needs
- [ ] Set up monitoring (metrics, tracing, logging)
- [ ] Validate configuration with validator
- [ ] Test with dry run

### Support

For configuration issues:

1. Check logs: `logs/rese-adapter.log`
2. Run validator: `python -m glue.adapters.rese_integration.config_validator`
3. Check documentation: This README_CONFIG.md
4. Open issue: [GitHub Issues](https://github.com/your-org/rese-adapter/issues)

### References

- [CLAUDE.md](../../../../CLAUDE.md) - Overall architecture principles
- [ADR.md](ADR.md) - Architecture Decision Records
- [API.md](API.md) - API documentation (if available)

# Adaptive MDAP - Troubleshooting Guide

> **Version**: 1.0.0  
> **Date**: February 2, 2026

---

## Common Issues and Solutions

### Issue 1: Adaptive MDAP Not Available

**Symptom:**
```
[WARNING] Adaptive MDAP not available - using standard allocation
```

**Causes:**
- Missing dependencies
- Import errors
- Virtual environment issues

**Solutions:**

1. **Install Dependencies:**
```bash
pip install -e .[adaptive]
```

2. **Check Imports:**
```python
python -c "from adaptive_mdap import TaskComplexityClassifier; print('OK')"
```

3. **Verify Virtual Environment:**
```bash
which python
pip list | grep adaptive
```

---

### Issue 2: Classification Timeout

**Symptom:**
```
[ERROR] Classification timeout after 5000ms
```

**Causes:**
- Large text input
- Slow embedding model
- Cache miss

**Solutions:**

1. **Reduce Text Length:**
```python
# Truncate long descriptions
sp = SubProblem(
    id="sp-001",
    description=description[:500],  # Limit to 500 chars
    ...
)
```

2. **Preload Embedding Model:**
```python
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier

# Initialize early
classifier = TaskComplexityClassifier()
classifier._ensure_model_loaded()
```

3. **Increase Timeout:**
```python
import os
os.environ["ADAPTIVE_MDAP_TIMEOUT"] = "10000"  # 10 seconds
```

---

### Issue 3: High Memory Usage

**Symptom:**
```
[WARNING] High memory usage detected
```

**Causes:**
- Large embedding cache
- Memory leaks
- Too many cached entries

**Solutions:**

1. **Clear Cache:**
```python
from adaptive_mdap.utils.cache import clear_cache, get_cache_stats

# Check current usage
stats = get_cache_stats()
print(f"Cache size: {stats}")

# Clear if needed
clear_cache()
```

2. **Limit Cache Size:**
```python
from adaptive_mdap.utils.cache import EmbeddingCache

cache = EmbeddingCache(max_size=1000)  # Limit entries
```

3. **Use LRU Cache:**
```python
# Configure in config
ADAPTIVE_MDAP_CACHE_STRATEGY=lru
ADAPTIVE_MDAP_CACHE_SIZE=500
```

---

### Issue 4: Slow Allocation Performance

**Symptom:**
```
[WARNING] Allocation took 50ms (target: <1ms)
```

**Causes:**
- Complex allocation logic
- Database queries
- Network delays

**Solutions:**

1. **Use Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def allocate_for_complexity(score: float):
    return allocator.allocate_resources(score)
```

2. **Profile Performance:**
```python
import time

start = time.time()
config = allocator.allocate_resources(score)
print(f"Allocation took {(time.time() - start)*1000:.2f}ms")
```

3. **Check for Blocking Operations:**
```python
# Ensure no I/O in allocation path
# Use pre-computed thresholds
```

---

### Issue 5: Incorrect Complexity Scores

**Symptom:**
```
Simple task scored 0.9 (should be ~0.2)
```

**Causes:**
- Embedding model mismatch
- Domain not recognized
- Feature weights misconfigured

**Solutions:**

1. **Check Embedding Model:**
```python
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier

classifier = TaskComplexityClassifier()
print(classifier.config.embedding_model)
```

2. **Specify Domain:**
```python
sp = SubProblem(
    id="sp-001",
    description="...",
    domain="mathematics",  # Explicit domain
    ...
)
```

3. **Adjust Feature Weights:**
```python
from adaptive_mdap.config.profiles import ConfigProfile

profile = ConfigProfile(
    feature_weights={
        "text_length": 0.3,
        "domain_rarity": 0.2,
        "depth": 0.2,
        ...
    }
)
```

---

### Issue 6: Alert Flooding

**Symptom:**
```
Too many adaptive alerts generated
```

**Causes:**
- Low alert thresholds
- No deduplication
- High-frequency operations

**Solutions:**

1. **Adjust Alert Thresholds:**
```python
from alerting_system import create_adaptive_classification_alert

# Increase threshold
alert = create_adaptive_classification_alert(
    subproblem_id="sp-001",
    complexity_score=0.65,
    latency_ms=150.0,
    threshold_ms=200.0  # Higher threshold
)
```

2. **Use Deduplication:**
```python
from alerting_system import get_alert_manager

manager = get_alert_manager()
manager.notification_config.deduplication_window = 300  # 5 minutes
```

3. **Batch Alerts:**
```python
# Collect and batch alert generation
# Use summary alerts instead of individual
```

---

### Issue 7: Profile Not Working

**Symptom:**
```
Profile 'aggressive' not applying correctly
```

**Causes:**
- Profile not loaded
- Default profile overriding
- Configuration conflict

**Solutions:**

1. **Verify Profile Loading:**
```python
from adaptive_mdap.config.profiles import load_profile

profile = load_profile("aggressive")
print(profile.__dict__)
```

2. **Check Configuration Priority:**
```python
# Environment variables override code config
import os
print(os.environ.get("ADAPTIVE_MDAP_PROFILE"))
```

3. **Explicit Profile Selection:**
```python
from adaptive_mdap import AdaptiveMDAPAllocator
from adaptive_mdap.config.profiles import load_profile

profile = load_profile("aggressive")
allocator = AdaptiveMDAPAllocator(profile=profile)
```

---

### Issue 8: Integration Not Working

**Symptom:**
```
Adaptive MDAP not triggering in workflow
```

**Causes:**
- Not enabled
- Missing imports
- Initialization order

**Solutions:**

1. **Check Enable Flag:**
```python
print(workflow_state.enable_adaptive_mdap)
print(workflow_state.metadata.get("adaptive_mdap_config"))
```

2. **Verify Imports:**
```python
from workflow_engine import ADAPTIVE_MDAP_AVAILABLE
print(f"Available: {ADAPTIVE_MDAP_AVAILABLE}")
```

3. **Check Session State:**
```python
import streamlit as st
print(st.session_state.get("enable_adaptive_mdap"))
```

---

## Debug Mode

### Enable Debug Logging

```python
import logging

logging.getLogger("adaptive_mdap").setLevel(logging.DEBUG)
```

### Verify Integration

```python
from workflow_engine import validate_adaptive_mdap_integration

results = validate_adaptive_mdap_integration()
print(f"Status: {results['status']}")
for check in results['checks']:
    print(f"  {check['name']}: {check['status']}")
```

### Check Metrics

```python
from monitoring_system import get_adaptive_metrics

metrics = get_adaptive_metrics()
print(json.dumps(metrics, indent=2))
```

---

## Performance Tuning

### Optimize Classification

```python
# Use faster embedding model
ADAPTIVE_MDAP_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Enable caching
ADAPTIVE_MDAP_CACHE_ENABLED=true
```

### Optimize Allocation

```python
# Pre-compute common allocations
from adaptive_mdap import AdaptiveMDAPAllocator

allocator = AdaptiveMDAPAllocator()
common_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
precomputed = {s: allocator.allocate_resources(s) for s in common_scores}
```

---

## Getting Help

1. **Check Documentation:**
   - `ADAPTIVE_MDAP_INTEGRATION_GUIDE.md`
   - `ADAPTIVE_MDAP_QUICK_START.md`

2. **Run Verification:**
   ```bash
   python check_wiring_complete.py
   ```

3. **Run Tests:**
   ```bash
   python test_adaptive_mdap_integration.py
   ```

4. **Check Logs:**
   ```bash
   tail -f logs/adaptive_mdap.log
   ```

---

**Still having issues?** Check the integration guide or run the verification script.

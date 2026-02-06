#!/usr/bin/env python
"""Test script for adaptive_mdap import fixes."""

import sys
import os

# Set up path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core-projects'))

# Block problematic imports that cause timeouts
for mod in ['knowledge_engine', 'openevolve', 'z3prover_integration', 
            'leanaide_continuous_math', 'integrations', 'config']:
    sys.modules[mod] = type(sys)(mod)

results = []

# Test 1: utils.logger
try:
    from adaptive_mdap.utils.logger import get_logger
    results.append(('utils.logger.get_logger', 'OK'))
except Exception as e:
    results.append(('utils.logger.get_logger', f'FAIL: {e}'))

# Test 2: utils.metrics
try:
    from adaptive_mdap.utils.metrics import get_metrics, MetricsCollector
    results.append(('utils.metrics.get_metrics', 'OK'))
except Exception as e:
    results.append(('utils.metrics.get_metrics', f'FAIL: {e}'))

# Test 3: utils.cache
try:
    from adaptive_mdap.utils.cache import EmbeddingCache, FeatureCache, get_cache_stats
    results.append(('utils.cache', 'OK'))
except Exception as e:
    results.append(('utils.cache', f'FAIL: {e}'))

# Test 4: utils.__init__ exports
try:
    from adaptive_mdap.utils import get_logger, get_metrics, EmbeddingCache, FeatureCache, get_cache_stats
    results.append(('utils.__init__ exports', 'OK'))
except Exception as e:
    results.append(('utils.__init__ exports', f'FAIL: {e}'))

# Test 5: config.profiles
try:
    from adaptive_mdap.config.profiles import ConfigProfile, get_profile_config, load_profile
    results.append(('config.profiles', 'OK'))
except Exception as e:
    results.append(('config.profiles', f'FAIL: {e}'))

# Test 6: classifiers
try:
    from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier, ClassifierConfig
    results.append(('classifiers.task_complexity_classifier', 'OK'))
except Exception as e:
    results.append(('classifiers.task_complexity_classifier', f'FAIL: {e}'))

# Test 7: monitoring.health
try:
    from adaptive_mdap.monitoring.health import HealthChecker, HealthCheckResult, ComponentStatus
    results.append(('monitoring.health', 'OK'))
except Exception as e:
    results.append(('monitoring.health', f'FAIL: {e}'))

# Test 8: monitoring.dashboard
try:
    from adaptive_mdap.monitoring.dashboard import DashboardGenerator, DashboardPanel
    results.append(('monitoring.dashboard', 'OK'))
except Exception as e:
    results.append(('monitoring.dashboard', f'FAIL: {e}'))

# Test 9: monitoring.alerts
try:
    from adaptive_mdap.monitoring.alerts import AlertingEngine, Alert, AlertRule, AlertSeverity
    results.append(('monitoring.alerts', 'OK'))
except Exception as e:
    results.append(('monitoring.alerts', f'FAIL: {e}'))

# Print results
print("\n" + "="*60)
print("ADAPTIVE_MDAP IMPORT TEST RESULTS")
print("="*60)
for name, status in results:
    symbol = "✓" if status == "OK" else "✗"
    print(f"{symbol} {name}: {status}")
print("="*60)

# Summary
ok_count = sum(1 for _, s in results if s == 'OK')
print(f"\nSummary: {ok_count}/{len(results)} tests passed")

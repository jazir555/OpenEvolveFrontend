# OpenEvolve-BubbleLab Security Report
# Top-Level Directory Only
# Generated: 2026-01-21 00:10:11

## Executive Summary

**Files Scanned:** 604
**Scan Scope:** Top-level directory ONLY (no subdirectories)

### Bandit Security Scanner Results

- **Security Issues Found:** 0
- **Files with Errors:** 0

**By Severity:**
- HIGH: 0
- MEDIUM: 0
- LOW: 0

### Custom Analysis Results

- **Syntax Errors:** 2
- **Bare Except Clauses:** 1
- **Try/Except/Pass Patterns:** 0
- **Pickle Usage:** 30
- **Hardcoded /tmp Paths:** 11

## Files with Most Issues

## Syntax Errors (Critical)

These files cannot be executed and must be fixed first:

### fix_tmp_paths.py
- **Line:** 13
- **Error:** unterminated string literal (detected at line 13)

### performance_optimization.py
- **Line:** 288
- **Error:** expected an indented block after 'with' statement on line 287

## Bare Except Clauses

Generic exception handlers that catch everything:

**edge_case_detector_fixed.py:185**
```python
if not in_try_except:
```

## Pickle Usage (Security Risk)

Insecure deserialization - should use JSON instead:

**advanced_unit_tests_comprehensive.py:32**
```python
# import pickle  # REMOVED - security risk
```

**auto_fix_security.py:159**
```python
# Check for pickle.load (B301)
```

**auto_fix_security.py:165**
```python
logger.critical(f"  [{self.filename}] pickle.load() at line {node.lineno} - MANUAL FIX REQUIRED")
```

**auto_fix_security.py:169**
```python
'fix': 'CRITICAL: Replace pickle.load() with json.load() - MANUAL FIX REQUIRED'
```

**auto_fix_security.py:325**
```python
# Check for pickle import or usage
```

**auto_fix_security.py:326**
```python
if 'pickle' in line and ('import' in line or 'pickle.' in line):
```

**auto_fix_security.py:445**
```python
logger.info(f"  - Replace pickle.load() with json.load(): {total_issues['pickle_usage']}")
```

**auto_fix_top_level.py:403**
```python
if 'pickle' in line and ('import' in line or 'pickle.' in line):
```

**blue_team_coordinator.py:31**
```python
# import pickle  # REMOVED - security risk
```

**evaluator_team_coordinator.py:37**
```python
# import pickle  # REMOVED - security risk
```

**fix_manual_security_issues.py:67**
```python
if 'pickle.load' in line:
```

**fix_manual_security_issues.py:69**
```python
elif 'pickle.dump' in line:
```

**fix_manual_security_issues.py:71**
```python
elif 'import pickle' in line:
```

**fix_manual_security_issues.py:227**
```python
report_lines.append("import pickle")
```

**fix_manual_security_issues.py:229**
```python
report_lines.append("    data = pickle.load(f)  # Can execute arbitrary code!")
```

**future_enhancements.py:26**
```python
import pickle
```

**future_enhancements.py:207**
```python
pickle.dump(model_data, f)
```

**future_enhancements.py:212**
```python
model_data = pickle.load(f)
```

**leanaide_mdap.py:39**
```python
# import pickle  # REMOVED - security risk
```

**llm_caching.py:9**
```python
#import pickle  # REMOVED - security risk
```

**mcts_evolved_policies.py:44**
```python
# import pickle  # REMOVED - security risk
```

**mcts_evolved_policies_mdap.py:36**
```python
# import pickle  # REMOVED - security risk
```

**red_team_coordinator.py:33**
```python
# import pickle  # REMOVED - security risk
```

**scan_top_level_only.py:122**
```python
if 'pickle' in line and ('import' in line or 'pickle.' in line):
```

**scan_top_level_only.py:350**
```python
report_lines.append("import pickle")
```

**scan_top_level_only.py:351**
```python
report_lines.append("data = pickle.load(open('data.pkl', 'rb'))")
```

**test_guardrails_integration.py:259**
```python
"os.system", "subprocess", "pickle.loads"
```

**validate_phase1_complete.py:147**
```python
# import pickle  # REMOVED - security risk
```

**workflow_enhanced_stages.py:1232**
```python
if "pickle.load" in content_lower or "marshal.load" in content_lower:
```

**workflow_enhanced_stages.py:1789**
```python
(r"pickle\.load", "Unsafe deserialization - pickle.load detected", 0.2),
```

## Hardcoded Temp Paths

Predictable temp directories - should use tempfile module:

**auto_fix_security.py:177**
```python
if isinstance(node.args[0].value, str) and '/tmp/' in node.args[0].value:
```

**auto_fix_security.py:330**
```python
if "'/tmp/" in line or '"/tmp/' in line:
```

**auto_fix_top_level.py:296**
```python
# TODO: Replace hardcoded /tmp with tempfile.mkdtemp() - needs_import = ('/tmp/' in content or '/tmp"' in content or "/tmp'" in content) and \
```

**auto_fix_top_level.py:297**
```python
needs_import = ('/tmp/' in content or '/tmp"' in content or "/tmp'" in content) and \
```

**fix_manual_security_issues.py:109**
```python
if 'open(' in line and '/tmp/' in line:
```

**fix_manual_security_issues.py:111**
```python
elif '/tmp/' in line and '=' in line:
```

**fix_manual_security_issues.py:241**
```python
report_lines.append("temp_dir = '/tmp/myapp_data'")
```

**fix_tmp_paths.py:13**
```python
content = re.sub(r'"/tmp/([a-zA-Z0-9_]+)', r'tempfile.mkdtemp(prefix=\1_')', content)
```

**fix_tmp_paths.py:14**
```python
content = re.sub(r"'/tmp/([a-zA-Z0-9_]+)", r'tempfile.mkdtemp(prefix=\1_')', content)
```

**maker_engine.py:371**
```python
>>> store = FileCheckpointStore(path="/tmp/checkpoint.json")
```

**scan_top_level_only.py:365**
```python
report_lines.append("temp_dir = '/tmp/myapp'")
```

## Recommended Fixes

### 1. Syntax Errors (Highest Priority)

Fix syntax errors first - these files cannot be imported or executed.

### 2. High Severity Security Issues

Address 0 HIGH severity security issues.

### 3. Bare Except Clauses

Replace 1 bare except clauses with specific exception types.

```python
# Before
try:
    risky_operation()
except:
    pass

# After
import logging
logger = logging.getLogger(__name__)

try:
    risky_operation()
except (ValueError, TypeError) as e:
    logger.error(f"Expected error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

### 4. Pickle Usage

Replace 30 pickle usage with JSON.

```python
# Before (insecure)
import pickle
data = pickle.load(open('data.pkl', 'rb'))

# After (secure)
import json
data = json.load(open('data.json', 'r'))
```

### 5. Hardcoded Temp Paths

Replace 11 hardcoded /tmp paths with tempfile module.

```python
# Before (insecure)
temp_dir = '/tmp/myapp'

# After (secure)
import tempfile
temp_dir = tempfile.mkdtemp(prefix='myapp_')
```

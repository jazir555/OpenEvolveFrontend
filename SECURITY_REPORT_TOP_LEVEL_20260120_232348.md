# OpenEvolve-BubbleLab Security Report
# Top-Level Directory Only
# Generated: 2026-01-20 23:23:48

## Executive Summary

**Files Scanned:** 596
**Scan Scope:** Top-level directory ONLY (no subdirectories)

### Bandit Security Scanner Results

- **Security Issues Found:** 0
- **Files with Errors:** 0

**By Severity:**
- HIGH: 0
- MEDIUM: 0
- LOW: 0

### Custom Analysis Results

- **Syntax Errors:** 12
- **Bare Except Clauses:** 130
- **Try/Except/Pass Patterns:** 0
- **Pickle Usage:** 51
- **Hardcoded /tmp Paths:** 11

## Files with Most Issues

## Syntax Errors (Critical)

These files cannot be executed and must be fixed first:

### ace_mcp_tools_FIXED.py
- **Line:** 262
- **Error:** invalid syntax

### adversarial_adapter.py
- **Line:** 355
- **Error:** expected 'except' or 'finally' block

### adversarial_error_handling.py
- **Line:** 778
- **Error:** 'await' outside function

### bubblelabs_evolution_integration.py
- **Line:** 449
- **Error:** expected 'except' or 'finally' block

### demo_mcts_mdap.py
- **Line:** 604
- **Error:** f-string expression part cannot include a backslash

### hybrid_error_handling.py
- **Line:** 297
- **Error:** 'await' outside function

### leanaide_mdap_demo.py
- **Line:** 44
- **Error:** unterminated string literal (detected at line 44)

### leanaide_sop_integration.py
- **Line:** 162
- **Error:** invalid syntax

### openevolve_leanaide_bridge.py
- **Line:** 483
- **Error:** invalid syntax

### simple_verify_implementation.py
- **Line:** 77
- **Error:** expected 'except' or 'finally' block

### sovereign_gauntlets.py
- **Line:** 451
- **Error:** expected an indented block after 'except' statement on line 449

### workflow_stage_functions.py
- **Line:** 90
- **Error:** unterminated string literal (detected at line 90)

## Bare Except Clauses

Generic exception handlers that catch everything:

**advanced_features.py:147**
```python
except:
```

**advanced_system_unit_tests.py:92**
```python
except:
```

**advanced_system_unit_tests.py:435**
```python
except:
```

**advanced_visualization.py:211**
```python
except:
```

**adversarial_performance.py:123**
```python
except:
```

**adversarial_performance.py:330**
```python
except:
```

**adversarial_performance.py:690**
```python
except:
```

**adversarial_performance.py:702**
```python
except:
```

**adversarial_realtime.py:540**
```python
except:
```

**analyze_bubbles.py:63**
```python
except:
```

**base_configuration.py:147**
```python
except:
```

**blue_team_utilities.py:1665**
```python
except:
```

**bubblelab-auto-setup-v1-backup.py:321**
```python
except:
```

**bubblelab-auto-setup-v1-backup.py:329**
```python
except:
```

**bubblelab-auto-setup-v2.py:367**
```python
except:
```

**bubblelab-auto-setup-v2.py:373**
```python
except:
```

**bubblelab-auto-setup-v2.py:412**
```python
except:
```

**bubblelab-auto-setup-v2.py:755**
```python
except:
```

**bubblelab-auto-setup-v2.py:763**
```python
except:
```

**bubblelab-auto-setup-v3.py:712**
```python
except:
```

**bubblelab-auto-setup-v3.py:891**
```python
except:
```

**bubblelab-auto-setup-v3.py:899**
```python
except:
```

**bubblelab-auto-setup.py:793**
```python
except:
```

**bubblelab-auto-setup.py:801**
```python
except:
```

**bubblelab-automation.py:128**
```python
except:
```

**bubblelab-automation.py:758**
```python
except:
```

**bubblelabs_ui_component.py:160**
```python
except:
```

**bubblelabs_ui_component.py:616**
```python
except:
```

**bubblelabs_ui_component.py:637**
```python
except:
```

**compare_parameter_managers.py:74**
```python
except:
```

*... and 100 more*

## Pickle Usage (Security Risk)

Insecure deserialization - should use JSON instead:

**advanced_cache.py:18**
```python
import pickle
```

**advanced_cache.py:127**
```python
size = sys.getsizeof(pickle.dumps(value))
```

**advanced_cache.py:311**
```python
value = pickle.loads(value_blob)
```

**advanced_cache.py:339**
```python
value_blob = pickle.dumps(value)
```

**advanced_unit_tests_comprehensive.py:32**
```python
import pickle
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

**blue_team_coordinator.py:31**
```python
import pickle
```

**blue_team_coordinator.py:966**
```python
pickle.dump(state, f)
```

**blue_team_coordinator.py:979**
```python
state = pickle.load(f)
```

**evaluator_team_coordinator.py:37**
```python
import pickle
```

**evaluator_team_coordinator.py:1662**
```python
pickle.dump(state, f)
```

**evaluator_team_coordinator.py:1673**
```python
state = pickle.load(f)
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
import pickle
```

**leanaide_mdap.py:1871**
```python
pickle.dump(checkpoint_data, f)
```

**leanaide_mdap.py:1893**
```python
checkpoint_data = pickle.load(f)
```

**llm_cache.py:14**
```python
import pickle
```

**llm_cache.py:66**
```python
cache_data = pickle.load(f)
```

*... and 21 more*

## Hardcoded Temp Paths

Predictable temp directories - should use tempfile module:

**add_class_function_docstrings.py:220**
```python
>>> store = FileCheckpointStore(base_path="/tmp/checkpoints")
```

**auto_fix_security.py:177**
```python
if isinstance(node.args[0].value, str) and '/tmp/' in node.args[0].value:
```

**auto_fix_security.py:330**
```python
if "'/tmp/" in line or '"/tmp/' in line:
```

**deployment_operations.py:285**
```python
tar.extractall(path='/tmp/sovereign_restore')
```

**deployment_operations.py:288**
```python
backup_db = '/tmp/sovereign_restore/database.db'
```

**deployment_operations.py:294**
```python
backup_config = '/tmp/sovereign_restore/config'
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

**maker_engine.py:362**
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

Replace 130 bare except clauses with specific exception types.

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

Replace 51 pickle usage with JSON.

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

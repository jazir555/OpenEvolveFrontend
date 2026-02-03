# Claudiomiro Integration Fix

## Problem

The Claudiomiro-CrewAI bridge was returning stub results because the Claudiomiro CLI was not being detected on Windows systems. This caused all 6 phase methods to return placeholder results instead of executing actual autonomous development operations.

## Root Causes

### 1. Windows CLI Detection Failure
On Windows, npm-installed CLI tools are installed as `.cmd` files (e.g., `claudiomiro.cmd`), but Python's `subprocess.run()` cannot execute them without either:
- Using the full path with `.cmd` extension
- Setting `shell=True` in subprocess calls

**Original Detection Code:**
```python
try:
    result = subprocess.run(
        ["claudiomiro", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        CLAUDIOMIRO_AVAILABLE = True
except FileNotFoundError:
    CLAUDIOMIRO_IMPORT_ERROR = "claudiomiro CLI not found in PATH"
```

This failed with `FileNotFoundError` on Windows even though `claudiomiro` was installed.

### 2. Import Errors
The `claudiomiro_crewai_bridge.py` was importing non-existent classes from `crewai_zero_error_workflow.py`:

**Incorrect Imports:**
```python
from crewai_zero_error_workflow import (
    CrewAIZeroErrorWorkflow,  # Doesn't exist
    ZeroErrorConfig,           # Doesn't exist
    create_zero_error_workflow,  # Doesn't exist
    create_zero_error_config,  # Doesn't exist
)
```

**Correct Imports:**
```python
from crewai_zero_error_workflow import (
    ZeroErrorWorkflow,
    WorkflowDefinition,
    ExecutionContext,
)
```

### 3. Missing shell=True in Subprocess Calls
Even after detection, all subprocess calls to execute claudiomiro were missing `shell=True` on Windows:

```python
# Before (fails on Windows with .cmd files)
result = subprocess.run(
    cmd,
    cwd=working_dir,
    capture_output=True,
    text=True,
    timeout=3600,
)

# After (works on Windows)
use_shell = CLAUDIOMIRO_PATH.endswith('.cmd') or CLAUDIOMIRO_PATH.endswith('.bat')
result = subprocess.run(
    cmd,
    cwd=working_dir,
    capture_output=True,
    text=True,
    timeout=3600,
    shell=use_shell,
)
```

## Solution

### 1. Enhanced CLI Detection

Created a robust `_find_claudiomiro_cli()` function that:

1. **Checks Windows npm paths first:**
   - `%USERPROFILE%\AppData\Roaming\npm\claudiomiro.cmd`
   - `C:\Program Files\nodejs\claudiomiro.cmd`
   - Glob pattern for user-specific paths

2. **Uses platform-specific commands:**
   - Windows: `where claudiomiro`
   - Unix/Linux/macOS: `which claudiomiro`

3. **Falls back to shell=True detection:**
   - Attempts direct execution with shell=True
   - Works across all platforms

```python
def _find_claudiomiro_cli() -> tuple[bool, str | None, str | None]:
    """Find claudiomiro CLI across different platforms."""
    import platform
    import glob as glob_module

    system = platform.system()

    # On Windows, try npm installation path first
    if system == "Windows":
        npm_paths = [
            os.path.expanduser(r"~\AppData\Roaming\npm\claudiomiro.cmd"),
            r"C:\Program Files\nodejs\claudiomiro.cmd",
            r"C:\Users\*\AppData\Roaming\npm\claudiomiro.cmd",
        ]

        for npm_path in npm_paths:
            try:
                matches = glob_module.glob(npm_path)
                if matches:
                    npm_path = matches[0]

                if os.path.exists(npm_path):
                    # Test if it works
                    result = subprocess.run(
                        [npm_path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        shell=True  # CRITICAL: shell=True for .cmd files
                    )
                    if result.returncode == 0:
                        return True, npm_path, None
            except Exception:
                continue

    # Try platform-specific commands
    # ... (see full implementation)

    return False, None, "claudiomiro CLI not found..."
```

### 2. Fixed Imports

Updated `claudiomiro_crewai_bridge.py` to use correct imports:

```python
# Before (incorrect)
from crewai_zero_error_workflow import (
    CrewAIZeroErrorWorkflow,
    ZeroErrorConfig,
    create_zero_error_workflow,
    create_zero_error_config,
)

# After (correct)
from crewai_zero_error_workflow import (
    ZeroErrorWorkflow,
    WorkflowDefinition,
    ExecutionContext,
)
```

### 3. Added shell=True to All Subprocess Calls

Updated all 6 subprocess calls that execute claudiomiro:

1. **execute_claudiomiro_task** (line 359)
2. **decompose_task_with_claudiomiro** (line 463)
3. **fix_tests_with_claudiomiro** (line 574)
4. **fix_branch_with_claudiomiro** (line 667)
5. **execute_multi_repo_task** (line 834)
6. **set_claudiomiro_config** (line 897)

Each call now includes:
```python
use_shell = CLAUDIOMIRO_PATH.endswith('.cmd') or CLAUDIOMIRO_PATH.endswith('.bat')
result = subprocess.run(
    cmd,
    ...,
    shell=use_shell,
)
```

### 4. Moved Logger Configuration

Moved logger initialization before claudiomiro detection to prevent "name 'logger' is not defined" errors:

```python
# Logging configuration - must be before claudiomiro detection
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Claudiomiro Availability Detection
CLAUDIOMIRO_AVAILABLE, CLAUDIOMIRO_PATH, CLAUDIOMIRO_IMPORT_ERROR = _find_claudiomiro_cli()
if CLAUDIOMIRO_AVAILABLE:
    logger.info(f"Claudiomiro CLI detected at: {CLAUDIOMIRO_PATH}")
```

## Verification

### Before Fix
```python
>>> from claudiomiro_mcp_tools import CLAUDIOMIRO_AVAILABLE
>>> CLAUDIOMIRO_AVAILABLE
False
>>> CLAUDIOMIRO_IMPORT_ERROR
'claudiomiro CLI not found in PATH'
```

### After Fix
```python
>>> from claudiomiro_mcp_tools import CLAUDIOMIRO_AVAILABLE
INFO:claudiomiro_mcp_tools:Claudiomiro CLI detected at: C:\Users\mmeadow\AppData\Roaming\npm\claudiomiro.cmd
>>> CLAUDIOMIRO_AVAILABLE
True
>>> CLAUDIOMIRO_PATH
'C:\\Users\\mmeadow\\AppData\\Roaming\\npm\\claudiomiro.cmd'

>>> from claudiomiro_crewai_bridge import ClaudiomiroCrewAIWorkflowBridge
>>> bridge = ClaudiomiroCrewAIWorkflowBridge(working_dir='.', ai_provider='claude')
INFO:claudiomiro_crewai_bridge: Claudiomiro-CrewAI Bridge initialized (MIT-licensed)
>>> bridge.working_dir
'.'
>>> bridge.ai_provider
'claude'
```

## Integration Status

### Before Fix (Detection Failed)
- **CLAUDIOMIRO_AVAILABLE**: False
- **All 6 phases**: Return stub results
- **Error**: "Claudiomiro not available"

### After Fix (Detection Successful)
- **CLAUDIOMIRO_AVAILABLE**: True
- **CLI Path**: `C:\Users\mmeadow\AppData\Roaming\npm\claudiomiro.cmd`
- **All 6 phases**: Execute with actual Claudiomiro CLI
- **Subprocess calls**: Use `shell=True` on Windows

## Files Modified

1. **claudiomiro_mcp_tools.py**
   - Added `_find_claudiomiro_cli()` function
   - Moved logger configuration before detection
   - Updated all 6 subprocess calls with `shell=use_shell`

2. **claudiomiro_crewai_bridge.py**
   - Fixed imports from `crewai_zero_error_workflow`
   - No other changes needed (detection is in MCP tools module)

## Claudiomiro CLI Requirements

- **Platform**: Windows, macOS, Linux
- **Installation**: `npm install -g claudiomiro`
- **Version**: v2.8.82 (tested)
- **Supported AI Providers**:
  - claude
  - codex
  - gemini
  - deep-seek
  - glm

## Security Considerations

Using `shell=True` in subprocess calls can be dangerous if inputs are not properly validated. However, all inputs in the claudiomiro MCP tools are validated before subprocess execution:

- **task_id**: Regex validated (alphanumeric, hyphens, underscores only)
- **working_dir**: Path traversal prevention
- **prompt**: Shell injection pattern detection
- **ai_provider**: Whitelist validation
- **fix_command**: Safe character validation

## Troubleshooting

### Error: "claudiomiro CLI not found in PATH"

**Solution**: Install Claudiomiro globally via npm:
```bash
npm install -g claudiomiro
```

### Error: "FileNotFoundError: [WinError 2] The system cannot find the file specified"

**Solution**: The fix has been applied. Ensure you have the latest version of `claudiomiro_mcp_tools.py`.

### Bridge creation fails with ImportError

**Solution**: Ensure `crewai_zero_error_workflow.py` contains the correct classes:
```bash
python -c "from crewai_zero_error_workflow import ZeroErrorWorkflow; print('OK')"
```

## Related Documentation

- [Claudiomiro Documentation](https://github.com/claudiomiro/claudiomiro)
- [CrewAI Integration Guide](../CrewAI/CREWAI_INTEGRATION.md)
- [MCP Tools Specification](../MCP/MCP_TOOLS_SPEC.md)

## Status

- **Issue**: Claudiomiro CLI not detected on Windows
- **Root Cause**: Missing `.cmd` extension handling, wrong imports, missing shell=True
- **Fix**: Enhanced detection, fixed imports, added shell=True
- **Status**: ✅ RESOLVED
- **Date**: 2026-02-02

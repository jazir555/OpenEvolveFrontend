# LeanAide Setup Guide

Complete setup guide for LeanAide - Lean 4 integration with LLM-powered autoformalization.

## Quick Start

```bash
# 1. Check current status
python setup_lean4.py --check-only

# 2. Auto-install Lean 4 and mathlib4
python setup_lean4.py --auto-install

# 3. Verify installation
python setup_lean4.py --check-only
```

## Prerequisites

- Python 3.10+
- Internet connection
- OpenAI or Anthropic API key (for LLM features)

## Installation Steps

### Step 1: Install Python Dependencies

```bash
# Install required packages
pip install openai anthropic aiohttp

# Optional: Install all dependencies
pip install -r requirements.txt
```

### Step 2: Setup Lean 4 (Automated)

```bash
# Run automated setup
python setup_lean4.py --auto-install
```

This will:
1. Install `elan` (Lean version manager)
2. Install Lean 4 stable toolchain
3. Setup mathlib4 project
4. Verify installation

### Step 3: Setup Environment Variables

```bash
# For OpenAI (recommended)
export OPENAI_API_KEY="sk-..."

# For Anthropic (alternative)
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Set preferred provider
export LLM_PROVIDER="openai"  # or "anthropic"
```

Add to your `~/.bashrc` or `~/.zshrc` for persistence.

### Step 4: Verify Setup

```bash
# Run verification
python lean4_integration_enhanced.py
```

Expected output:
```
============================================================
Lean 4 Enhanced Integration - LLM-Powered Autoformalization
============================================================

📊 Service Status:
  Lean available: True
  Lake available: True
  LLM available:  True (openai)

1. VERIFY LEAN 4 CODE
----------------------------------------
   Status: success
   Success: True
   Errors: None

2. LLM AUTOFORMALIZATION
----------------------------------------
   Input: The limit as x approaches 0 of sin(x)/x equals 1
   Success: True
   Confidence: 0.85
   LLM Provider: openai
   Generated Code:
   import Mathlib
   ...
```

## Manual Installation (if auto-install fails)

### Linux/macOS

```bash
# 1. Install elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# 2. Install Lean 4
elan toolchain install stable
elan default stable

# 3. Verify
lean --version  # Should show Lean 4.x
lake --version  # Should show Lake version

# 4. Setup mathlib4 project
mkdir -p ~/lean_projects
cd ~/lean_projects
lake new my_project math
cd my_project
lake update
lake build
```

### Windows

1. Download elan installer from:
   https://github.com/leanprover/elan/releases/latest

2. Run the installer and follow prompts

3. Open new PowerShell and run:
   ```powershell
   elan toolchain install stable
   elan default stable
   ```

4. Verify installation:
   ```powershell
   lean --version
   lake --version
   ```

### Docker Alternative

```bash
docker run -it --rm leanprovercommunity/lean4:latest
```

## Troubleshooting

### Issue: "lean: command not found"

**Solution:**
```bash
# Check if elan is installed
which elan || echo "elan not found"

# If elan exists but lean doesn't
source $HOME/.elan/env
export PATH="$HOME/.elan/bin:$PATH"
```

### Issue: "No LLM provider available"

**Solution:**
```bash
# Install OpenAI package
pip install openai

# Set API key
export OPENAI_API_KEY="your-key-here"

# Verify
echo $OPENAI_API_KEY
```

### Issue: Mathlib4 not found

**Solution:**
```bash
# Create a mathlib4 project
python setup_lean4.py --setup-mathlib --project-dir ./my_mathlib

# Or manually
cd ~/lean_projects
lake new my_mathlib math
lake update
lake build
```

### Issue: "Timeout during setup"

**Solution:**
Mathlib4 is large and takes time to download. Increase timeout:
```bash
# Set longer timeout
export LEAN_SETUP_TIMEOUT=1200  # 20 minutes

# Or manually setup
lake update  # Run separately
lake build   # Run separately
```

### Issue: Permission denied on macOS

**Solution:**
```bash
# If you get "cannot be opened because the developer cannot be verified"
# Go to System Preferences > Security & Privacy > General
# Click "Allow Anyway" next to the lean message

# Or via command line:
sudo xattr -rd com.apple.quarantine ~/.elan/bin/lean
```

## Testing

### Run Basic Tests

```bash
# Test Lean 4 installation
python -c "
from setup_lean4 import detect_lean_installation
status = detect_lean_installation()
print(f'Lean: {status.lean_available}')
print(f'Lake: {status.lake_available}')
print(f'Mathlib: {status.mathlib_available}')
"
```

### Run Integration Tests

```bash
# Test with LLM (requires API key)
python lean4_integration_enhanced.py

# Test without LLM (basic functionality)
python lean4_integration.py
```

### Run Full Test Suite

```bash
# Run all LeanAide tests
pytest test_leanaide_continuous_math.py -v

# Run specific test
pytest test_leanaide_continuous_math.py::TestLean4Integration -v
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `LEAN_EXECUTABLE` | Path to lean executable | `lean` |
| `LAKE_EXECUTABLE` | Path to lake executable | `lake` |
| `LEAN_TIMEOUT` | Verification timeout (seconds) | `60` |

### Python Configuration

```python
from lean4_integration_enhanced import (
    LeanAideServiceEnhanced,
    Lean4ServerConfig,
    LLMProvider
)

# Configure with explicit API keys
config = Lean4ServerConfig(
    openai_api_key="sk-...",
    openai_model="gpt-4",
    llm_provider=LLMProvider.OPENAI,
    timeout_seconds=120
)

service = LeanAideServiceEnhanced(config)
```

## Verification Checklist

- [ ] `python setup_lean4.py --check-only` shows Lean available
- [ ] `lean --version` works in terminal
- [ ] `lake --version` works in terminal
- [ ] OpenAI/Anthropic API key is set
- [ ] `python lean4_integration_enhanced.py` runs successfully
- [ ] Tests pass: `pytest test_leanaide_continuous_math.py -v`

## Getting Help

1. Check Lean 4 documentation: https://lean-lang.org/lean4/doc/
2. Mathlib4 documentation: https://leanprover-community.github.io/mathlib4_docs/
3. Run diagnostics: `python setup_lean4.py --check-only --json`
4. View setup instructions: `python setup_lean4.py --instructions`

## Next Steps

After setup, explore:

1. **Basic Verification**: Test Lean code compilation
2. **Autoformalization**: Convert natural language to Lean
3. **Proof Completion**: Use LLM to complete proofs
4. **Integration**: Use with OpenEvolve workflows

```python
# Example: Basic usage
import asyncio
from lean4_integration_enhanced import create_lean4_service

async def main():
    service = create_lean4_service(openai_api_key="sk-...")
    
    # Verify code
    result = await service.verify("theorem t : 1+1=2 := by rfl")
    print(f"Verification: {result.success}")
    
    # Autoformalize
    result = await service.autoformalize(
        "The square root of 2 is irrational"
    )
    print(f"Generated:\n{result.lean_code}")

asyncio.run(main())
```

---

**Note**: This setup guide covers the enhanced LeanAide integration with real LLM support and automatic Lean 4 installation detection.

# Bug Fix Adapters for Core Projects

This directory contains **glue code adapters** that fix bugs in core projects without modifying the core project files themselves.

## 📐 Architecture Principles

All fixes follow the **Anti-Corruption Layer** pattern from CLAUDE.md:

- ✅ **DO NOT modify core project files**
- ✅ **Wrap/core classes/functions** in adapters
- ✅ **Provide corrected behavior** from the glue layer
- ✅ **Maintain isolation** between projects

## 🚫 Core Projects (Immutable - DO NOT EDIT)

- **crewai** - Task management system
- **openevolve** - Evolutionary coding framework
- **roma** - Decomposition system
- **bubblelab** - Enterprise platform
- **datapizza** - Multi-agent system
- **claudiomiro** - Decomposition integration
- **graphiti** - Knowledge graph
- **global-chem** - Chemical informatics
- **deep-research-agent** - Research agent
- **All kg/\*** - Knowledge graph projects
- **leanaide** - Formal verification
- **curie** - Analytics
- **PAMI** - Pattern mining
- **research-quest** - Research system
- **ragbits** - RAG system
- **steer** - Verification
- **Agentic Context Engine (ACE)** - Agent framework
- **uqsa** - Uncertainty quantification

## 📦 Bug Fix Adapters

### 1. CrewAIConfigOverride

**Fixes invalid paths in CrewAI config:**
- `phases_folder`: `./example_workflows/crackme_solving` → `./example_workflows/prd_to_software`
- `worktree_base`: `/tmp/crewai_worktrees` → `./crewai_worktrees`
- `project_root`: `/tmp/test_3gaur34` → `.`
- `main_repo_path`: `/tmp/test_3gaur34` → `.`

**Usage:**
```python
from integrations.bug_fixes import CrewAIConfigOverride

# Get fixed config
config_override = CrewAIConfigOverride()
fixed_config = config_override.get_fixed_config()

# Use with CrewAI
from CrewAI.src.core.simple_config import load_config
original_config = load_config()
fixed_config = config_override.apply_fixes(original_config)
```

### 2. EvolutionConfigurationWrapper

**Fixes duplicate dataclass fields in evolution.py:**
- Lines 84-91: Original field definitions
- Lines 92-96: **Duplicates** (cause overwrites)
- Wrapper ensures only first occurrence is used

**Duplicate Fields:**
- `convergence_threshold` (lines 84, 92)
- `fitness_function` (lines 85, 93)
- `elitism` (lines 89, 94)
- `diversity_maintenance` (lines 90, 95)
- `adaptive_parameters` (lines 91, 96)

**Usage:**
```python
from integrations.bug_fixes import EvolutionConfigurationWrapper

# Use wrapper instead of original class
config = EvolutionConfigurationWrapper(
    evolution_mode="standard",
    max_iterations=100
)

# Access wrapped config
evolution_config = config.get_config()

# Validate
issues = config.validate()
if issues:
    print(f"Config issues: {issues}")
```

### 3. AdversarialImportResolver

**Fixes circular import in adversarial system:**
- `adversarial_maker_integration.py` → `adversarial.py` → `red_team.py`
- Line 244: `RedTeamStrategy = RedTeamStrategy.ADVERSARIAL` fails when imports return None

**Usage:**
```python
from integrations.bug_fixes import AdversarialImportResolver, RedTeamStrategyProxy

# Method 1: Use resolver
resolver = AdversarialImportResolver()
RedTeamStrategy = resolver.get_red_team_strategy()
strategy = resolver.get_default_strategy()  # Safe fallback

# Method 2: Use proxy in default arguments
def my_function(attack_method=RedTeamStrategyProxy.DEFAULT):
    attack_method = RedTeamStrategyProxy.resolve(attack_method)
    # Use attack_method...

# Method 3: Patch at startup (temporary workaround)
from integrations.bug_fixes.adversarial_import_resolver import patch_adversarial_maker_init
patch_adversarial_maker_init()
```

### 4. ConfigProvider

**Provides configuration management without editing core files:**

**Features:**
- Auto-generates `.env` file with secure keys
- Validates required configuration
- Provides environment variable access
- Creates required directories

**Usage:**
```python
from integrations.bug_fixes import ConfigProvider

# Setup configuration
provider = ConfigProvider()
provider.setup_env()  # Creates .env in integrations/bug_fixes/
provider.ensure_directories()  # Creates crewai_worktrees, etc.

# Load into environment
provider.load_dotenv()

# Validate
issues = provider.validate_config()
if issues:
    for issue in issues:
        print(f"Config issue: {issue}")

# Quick access
api_key = provider.get_env('OPENAI_API_KEY')
```

**Quick Setup:**
```python
from integrations.bug_fixes.config_provider import setup_config

provider = setup_config(force=False)  # Don't overwrite existing
provider.load_dotenv()
```

## 🚀 Quick Start

### Apply All Bug Fixes

```python
# At application startup
from integrations.bug_fixes import (
    ConfigProvider,
    CrewAIConfigOverride,
    EvolutionConfigurationWrapper,
    AdversarialImportResolver
)

# 1. Setup configuration
config_provider = ConfigProvider()
config_provider.setup_env()
config_provider.load_dotenv()
config_provider.ensure_directories()

# 2. Validate configuration
issues = config_provider.validate_config()
if issues:
    print("Configuration issues:")
    for issue in issues:
        print(f"  - {issue}")

# 3. Apply CrewAI config fix
crewai_override = CrewAIConfigOverride()
crewai_config = crewai_override.get_fixed_config()

# 4. Use Evolution configuration wrapper
evolution_config = EvolutionConfigurationWrapper(
    max_iterations=100,
    population_size=20
)

# 5. Patch adversarial import issues (optional)
from integrations.bug_fixes.adversarial_import_resolver import patch_adversarial_maker_init
patch_adversarial_maker_init()
```

### Individual Bug Fixes

```python
# Fix just CrewAI config
from integrations.bug_fixes import CrewAIConfigOverride
config = CrewAIConfigOverride().get_fixed_config()

# Fix just Evolution config
from integrations.bug_fixes import EvolutionConfigurationWrapper
config = EvolutionConfigurationWrapper(max_iterations=100)

# Fix just adversarial imports
from integrations.bug_fixes import RedTeamStrategyProxy
def __init__(self, attack_method=RedTeamStrategyProxy.DEFAULT):
    attack_method = RedTeamStrategyProxy.resolve(attack_method)
```

## 📝 Configuration Files

### .env File Location

The `.env` file is created in:
```
integrations/bug_fixes/.env
```

**NOT** in the root directory (to avoid modifying core projects).

### Environment Variables

**Required:**
- `SECRET_KEY` - Auto-generated (32-byte hex)
- `KEY_ENCRYPTION_KEY` - Auto-generated (32-byte hex)

**Recommended:**
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

## 🧪 Testing

```python
# Test bug fixes
from integrations.bug_fixes import (
    CrewAIConfigOverride,
    EvolutionConfigurationWrapper,
    AdversarialImportResolver,
    ConfigProvider
)

# Test 1: CrewAI config
crewai_config = CrewAIConfigOverride().get_fixed_config()
assert crewai_config['paths']['worktree_base'] == './crewai_worktrees'
print("✓ CrewAI config fix works")

# Test 2: Evolution wrapper
evolution_config = EvolutionConfigurationWrapper(max_iterations=100)
assert evolution_config.max_iterations == 100
issues = evolution_config.validate()
assert len(issues) == 0, f"Validation issues: {issues}"
print("✓ Evolution wrapper works")

# Test 3: Adversarial resolver
resolver = AdversarialImportResolver()
strategy = resolver.get_default_strategy()
assert strategy in ["ADVERSARIAL", resolver._red_team_strategy.ADVERSARIAL]
print("✓ Adversarial resolver works")

# Test 4: Config provider
provider = ConfigProvider()
provider.setup_env(force=True)
assert os.path.exists(provider.env_path)
print("✓ Config provider works")
```

## 🔍 How It Works

### Anti-Corruption Layer Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Bug Fix Adapters (This Directory)               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ CrewAIConfig │  │  EvolutionConfig │                 │
│  │     Override     │  │     Wrapper      │                 │
│  └──────────────────┘  └──────────────────┘                 │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  Adversarial     │  │   ConfigProvider │                 │
│  │ ImportResolver   │  │                  │                 │
│  └──────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Projects                             │
│  (crewai, openevolve, roma, etc.)                       │
│              DO NOT EDIT - IMMUTABLE                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Isolation** - Bug fixes live separately from core code
2. **Delegation** - Adapters wrap/core classes, not replace them
3. **Override** - Provide corrected values via config/wrappers
4. **Validation** - Check and log issues rather than crashing
5. **Documentation** - Every bug fix is clearly documented

## 📚 Related Documentation

- **CLAUDE.md** - The Federation Constitution (Architecture rules)
- **../INTEGRATION_GUIDE.md** - Integration patterns
- **../README.md** - Project overview

## 🤝 Contributing

When adding new bug fixes:

1. **Create a new adapter file** in this directory
2. **Follow the Anti-Corruption Layer pattern**
3. **Document the bug clearly** (file, line, description)
4. **Provide usage examples**
5. **Add tests** if applicable
6. **Update this README**

**Template:**
```python
"""
Bug Fix Adapter for [Project]

Bug Fixed:
- File: [project/file.py]
- Line: [line number]
- Issue: [description]

Solution:
- [How the adapter fixes it without modifying core]

Usage:
    from integrations.bug_fixes import [AdapterName]
    # Usage example
"""
```

## ⚠️ Important Notes

1. **Never edit core project files** - All fixes must be in this directory
2. **Test thoroughly** - Ensure adapters work correctly
3. **Document everything** - Future maintainers need to understand the fixes
4. **Version control** - Track changes to adapters (not core projects)
5. **Performance** - Adapters should add minimal overhead

## 📞 Support

For questions or issues:
1. Check this README first
2. Review the adapter's docstrings
3. Look at usage examples
4. Consult CLAUDE.md for architecture principles

---

**Generated:** 2026-01-07
**Author:** OpenEvolve Frontend Team
**Pattern:** Anti-Corruption Layer (CLAUDE.md compliant)

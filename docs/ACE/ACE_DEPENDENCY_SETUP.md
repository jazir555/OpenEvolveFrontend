# ACE Dependency Setup Guide

## Problem

The ACE (Agentic Context Engine) integration was failing with stub results because required dependencies were not installed. This caused all 6 phase methods to return placeholder results instead of executing actual learning operations.

## Root Cause

The `ace_crewai_bridge.py` module requires several Python packages that were not installed:
- `instructor>=1.0.0` - Required for ACE LiteLLMClient initialization
- `python-toon>=0.1.0` - Required for skillbook TOON compression

When these dependencies were missing:
1. ACE imports would succeed (package structure exists)
2. `_initialize_ace_components()` would fail
3. `agent`, `reflector`, and `skill_manager` would be `None`
4. All phase methods would return stub results

## Solution

### Quick Fix

Run the automated setup script:

```bash
python setup_ace_dependencies.py
```

### Manual Installation

Install all ACE core dependencies:

```bash
pip install litellm>=1.78.0 \
            pydantic>=2.0.0 \
            python-dotenv>=1.0.0 \
            python-toon>=0.1.0 \
            tenacity>=8.0.0 \
            instructor>=1.0.0
```

### Install from ACE Project

Alternatively, install ACE with all dependencies:

```bash
cd agentic-context-engine
pip install -e .
```

Or with optional dependencies:

```bash
pip install -e ".[all]"  # Includes observability, langchain, transformers
```

## Verification

Test that ACE is working:

```python
from ace_crewai_bridge import ACECrewAIWorkflowBridge

bridge = ACECrewAIWorkflowBridge(
    model='gpt-4o-mini',
    enable_learning=True
)

# Check components are initialized
print(f"Agent: {bridge.agent}")  # Should be: <ace.roles.Agent object>
print(f"Reflector: {bridge.reflector}")  # Should be: <ace.roles.Reflector object>
print(f"Skill Manager: {bridge.skill_manager}")  # Should be: <ace.roles.SkillManager object>

# Test skill injection
skills = bridge.inject_skills('test context')
print(f"Skills injected: {'LEARNED SKILLS' in skills}")  # Should be: True
```

Expected output:
```
Agent: <ace.roles.Agent object at 0x...>
Reflector: <ace.roles.Reflector object at 0x...>
Skill Manager: <ace.roles.SkillManager object at 0x...>
Skills injected: True
```

## Integration Status

### Before Fix (Missing Dependencies)
- **ACE_AVAILABLE**: False (imports succeed but initialization fails)
- **Agent/Reflector/SkillManager**: None
- **All 6 phases**: Return stub results
- **Learning**: Disabled

### After Fix (Dependencies Installed)
- **ACE_AVAILABLE**: True (implicit - components initialize successfully)
- **Agent/Reflector/SkillManager**: Initialized
- **All 6 phases**: Execute with actual learning
- **Learning**: Enabled

## Dependencies Reference

From `agentic-context-engine/pyproject.toml`:

| Package | Version | Purpose | Required |
|---------|---------|---------|----------|
| litellm | >=1.78.0 | Unified LLM API (100+ providers) | Yes |
| pydantic | >=2.0.0 | Data validation | Yes |
| python-dotenv | >=1.0.0 | Environment variables | Yes |
| python-toon | >=0.1.0 | TOON compression (skillbook) | Yes |
| tenacity | >=8.0.0 | Retry logic | Yes |
| instructor | >=1.0.0 | Robust JSON parsing | Yes |
| browser-use | >=0.9.1 | Browser automation | Optional |
| opik | >=1.8.0 | Observability | Optional |
| langchain-* | >=0.2.0 | LangChain integration | Optional |
| transformers | >=4.30.0 | Local models | Optional |

## Troubleshooting

### Error: "No module named 'instructor'"

**Solution:**
```bash
pip install instructor>=1.0.0
```

### Error: "TOON compression requires python-toon"

**Solution:**
```bash
pip install python-toon>=0.1.0
```

### Error: "Failed to initialize ACE components"

**Cause**: Missing one or more core dependencies

**Solution**: Run the setup script:
```bash
python setup_ace_dependencies.py
```

### Bridge creates but components are None

**Symptoms:**
```python
bridge = ACECrewAIWorkflowBridge()
print(bridge.agent)  # None
print(bridge.reflector)  # None
print(bridge.skill_manager)  # None
```

**Solution**: Check logs for initialization errors:
```bash
# Look for: "Failed to initialize ACE components: ..."
# Install missing dependency based on error message
```

## API Key Configuration

After installing dependencies, configure your LLM API key:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-api-key"

# Or other providers
export ANTHROPIC_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-api-key"
```

## Next Steps

1. ✅ Install ACE dependencies (completed)
2. Configure API key for your LLM provider
3. Test ACE bridge with a simple example
4. Integrate ACE learning into your CrewAI workflows
5. Enable skillbook persistence for continuous learning

## Related Documentation

- [ACE Integration Guide](ACE_INTEGRATION_GUIDE.md)
- [ACE Critical Bugs Fixed](ACE_CRITICAL_BUGS_FIXED.md)
- [agentic-context-engine/README.md](../../agentic-context-engine/README.md)
- [agentic-context-engine/CLAUDE.md](../../agentic-context-engine/CLAUDE.md)

## Status

- **Issue**: ACE learning feedback loop returning stub results
- **Root Cause**: Missing `instructor` and `python-toon` dependencies
- **Fix**: Created setup script and documentation
- **Status**: ✅ RESOLVED
- **Date**: 2026-02-02

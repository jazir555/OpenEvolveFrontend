# ACE Systems Comparison: Custom ACE vs Kayba ACE

**Date:** 2025-02-03
**Purpose:** Clarify the two ACE implementations in the codebase

---

## TL;DR

You have **TWO different ACE systems** that implement the **same research paper** but serve different purposes:

| System | Location | Purpose | Status |
|--------|----------|---------|--------|
| **Custom ACE** | `/ace/` | Research implementation | Paper reproduction focus |
| **Kayba ACE** | `/core-projects/agentic-context-engine/` | Production framework | Enterprise-ready |

Both implement: **"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"** (arXiv:2510.04618)

---

## 1. Identity & Origins

### Custom ACE
- **Acronym:** ACE = Agentic Context Engineering
- **Full Name:** ACE (Agent-Curator-Environment)
- **Institution:** Stanford University & SambaNova Systems
- **Repository:** https://github.com/ace-agent/ace
- **Primary Use:** Research experiments, benchmarking, paper reproduction

### Kayba ACE
- **Acronym:** ACE = Agentic Context Engine
- **Full Name:** Agentic Context Engine (ACE)
- **Organization:** Kayba AI
- **PyPI Package:** `ace-framework`
- **Primary Use:** Production deployments, framework integrations

### Key Point
**They are NOT the same codebase.** They are independent implementations of the same research paper.

---

## 2. Architecture Comparison

### Custom ACE: Generator → Reflector → Curator

```
┌─────────────────────────────────────────────────────┐
│              CUSTOM ACE (Research)                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐      ┌──────────────┐             │
│  │ GENERATOR    │─────▶│ REFLECTOR    │             │
│  │              │      │              │             │
│  │ - Produces   │      │ - Analyzes   │             │
│  │   answers    │      │   outcomes   │             │
│  │ - Uses       │      │ - Tags       │             │
│  │   playbook   │      │   bullets    │             │
│  └──────────────┘      └──────┬───────┘             │
│                               │                      │
│                               ▼                      │
│                      ┌──────────────┐               │
│                      │ CURATOR      │               │
│                      │              │               │
│                      │ - Updates    │               │
│                      │   playbook   │               │
│                      │ - Manages    │               │
│                      │   bullets    │               │
│                      └──────┬───────┘               │
│                             │                        │
│                             └──────► Playbook        │
└─────────────────────────────────────────────────────┘
```

**Knowledge Format:** "Playbook" with bullet points
```
[STRATEGIES & INSIGHTS]
[str-00001] helpful=5 harmful=0 :: Always verify data types
[str-00002] helpful=3 harmful=1 :: Check edge cases

[FORMULAS & CALCULATIONS]
[calc-00003] helpful=8 harmful=0 :: NPV = Σ(CF / (1+r)^t)
```

---

### Kayba ACE: Agent → Reflector → SkillManager

```
┌─────────────────────────────────────────────────────┐
│           KAYBA ACE (Production)                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────┐      ┌──────────────┐      ┌──────┐│
│  │  AGENT     │─────▶│ REFLECTOR    │─────▶│SKILL ││
│  │            │      │              │      │MGR   ││
│  │ - Generates│      │ - Analyzes   │      │      ││
│  │   answers  │      │   feedback   │      │-Updates││
│  │ - Cites    │      │ - Tags       │      │ skillbook││
│  │   skills   │      │   skills     │      │      ││
│  └────────────┘      └──────────────┘      └───┬──┘│
│       ▲                                          │  │
│       │                                          │  │
│       └──────────────────────────────────────────┘  │
│                    Skillbook (TOON format)          │
└─────────────────────────────────────────────────────┘
```

**Knowledge Format:** "Skillbook" with TOON (Token-Oriented Object Notation)
- 16-62% token savings vs standard format
- More efficient storage and retrieval

---

## 3. Feature Comparison Matrix

| Feature | Custom ACE | Kayba ACE |
|---------|-----------|-----------|
| **Core Paper** | arXiv:2510.04618 | arXiv:2510.04618 (same) |
| **Terminology** | Generator-Reflector-Curator | Agent-Reflector-SkillManager |
| **Knowledge Store** | Playbook (bullets) | Skillbook (skills + TOON) |
| **Training Modes** | Offline, Online, Eval-only | Offline, Online, **Async** |
| **Token Optimization** | Manual budget management | TOON format (16-62% savings) |
| **Observability** | Basic logging | **Opik integration** |
| **LLM Support** | SambaNova, Together, OpenAI | **100+ providers** (LiteLLM) |
| **Framework Integrations** | ❌ None | ✅ LangChain, browser-use, Claude Code |
| **Async Learning** | ❌ No | ✅ Yes (3x faster) |
| **Checkpoint System** | Basic | Advanced with intervals |
| **PyPI Package** | ❌ No | ✅ `ace-framework` |
| **Production Ready** | Research-focused | ✅ Enterprise-ready |
| **Deduplication** | BulletpointAnalyzer | DeduplicationManager |
| **JSON Parsing** | Custom | **Instructor** (robust) |
| **Test Suite** | Basic | Comprehensive (pytest) |

---

## 4. Code Structure

### Custom ACE Structure
```
/ace/
├── ace/
│   ├── ace.py              # Main orchestrator (1142 lines)
│   ├── core/
│   │   ├── generator.py    # Answer generation
│   │   ├── reflector.py    # Analysis & tagging
│   │   ├── curator.py      # Playbook management
│   │   └── bulletpoint_analyzer.py
│   ├── prompts/            # Prompt templates
│   └── examples/           # Finance, AppWorld demos
├── README.md
└── EXTENDING_ACE.md
```

**Key File:** `ace.py` is a monolithic orchestrator with all logic in one file.

---

### Kayba ACE Structure
```
/core-projects/agentic-context-engine/
├── ace/
│   ├── __init__.py         # Main exports
│   ├── adaptation.py       # OfflineACE/OnlineACE (847 lines)
│   ├── roles.py            # Agent, Reflector, SkillManager
│   ├── skillbook.py        # Skill & Skillbook classes
│   ├── updates.py          # UpdateOperation system
│   ├── async_learning.py   # Parallel learning
│   ├── llm.py              # LLM client abstractions
│   ├── prompts_v2_1.py     # Production prompts
│   ├── integrations/       # Framework wrappers
│   │   ├── litellm.py      # ACELiteLLM
│   │   ├── langchain.py    # ACELangChain
│   │   ├── browser_use.py  # ACEAgent
│   │   └── claude_code.py  # ACEClaudeCode
│   ├── deduplication/      # Skill consolidation
│   └── observability/      # Opik monitoring
├── examples/
├── benchmarks/
└── README.md
```

**Key File:** `adaptation.py` orchestrates, but logic is modular across multiple files.

---

## 5. When to Use Which System

### Use Custom ACE When:

✅ **Research & Academic Work**
- Reproducing paper results exactly
- Benchmarking on AppWorld, FiNER, XBRL
- Understanding core algorithm

✅ **Financial Domain Tasks**
- Built-in finance data processors
- Pre-configured for financial benchmarks

✅ **Minimal Dependencies**
- Want simpler codebase
- Don't need framework integrations

✅ **Paper Validation**
- Need exact research implementation
- Comparing against baseline metrics

**Example:**
```python
from ace import ACE

ace_system = ACE(
    api_provider="sambanova",
    generator_model="DeepSeek-V3.1",
    reflector_model="DeepSeek-V3.1",
    curator_model="DeepSeek-V3.1"
)

results = ace_system.run(
    mode='offline',
    train_samples=finance_train,
    val_samples=finance_val
)
```

---

### Use Kayba ACE When:

✅ **Production Deployment**
- Need enterprise-grade reliability
- Require monitoring and observability

✅ **Framework Integration**
- Using LangChain chains
- Browser automation with browser-use
- CrewAI workflows
- Claude Code CLI enhancement

✅ **Performance Critical**
- Async learning (3x faster)
- TOON format efficiency
- 100+ LLM provider choice

✅ **Quick Prototyping**
- Simple API: `ACELiteLLM`
- Pre-built integrations
- Less boilerplate

**Example - Simple Start:**
```python
from ace import ACELiteLLM

agent = ACELiteLLM(model="gpt-4o-mini")
answer = agent.ask("What does ACE do?")
# Automatically learns
```

**Example - Browser Automation:**
```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

agent = ACEAgent(
    llm=ChatBrowserUse(),
    ace_model="gpt-4o-mini"
)
await agent.run(task="Find top Hacker News post")
agent.save_skillbook("expert.json")
```

**Example - LangChain:**
```python
from ace import ACELangChain

ace_chain = ACELangChain(runnable=your_langchain_chain)
result = ace_chain.invoke({"question": "Your task"})
```

---

## 6. Bridge Integration

The OpenEvolve project includes a bridge at `/ace_crewai_bridge.py`:

```python
# Lines 85-87
ACE_PATH = os.path.join(os.path.dirname(__file__), "agentic-context-engine")
if os.path.exists(ACE_PATH) and ACE_PATH not in sys.path:
    sys.path.insert(0, ACE_PATH)

# Lines 94-100
try:
    from ace import (
        Skillbook,
        Skill,
        Sample,
        SimpleEnvironment,
        OfflineACE,
        OnlineACE,
```

This allows:
1. **Switching between implementations**
2. **Validating results across both**
3. **Using Custom ACE for experiments, Kayba ACE for production**

---

## 7. Key Differentiators

### Custom ACE Strengths
- ✅ Exact research implementation
- ✅ Financial domain pre-configured
- ✅ Simpler codebase (easier to study)
- ✅ Paper reproduction validated

### Kayba ACE Strengths
- ✅ Production-ready with monitoring
- ✅ Framework integrations (LangChain, browser-use, CrewAI)
- ✅ Async learning (3x faster)
- ✅ TOON format (16-62% token savings)
- ✅ 100+ LLM providers
- ✅ Enterprise features (Opik, checkpoints, deduplication)

---

## 8. Decision Matrix

| Criteria | Weight | Custom ACE | Kayba ACE |
|----------|--------|------------|-----------|
| Research paper reproduction | ⭐⭐⭐⭐⭐ | ✅ Best | ❌ Diverged |
| Production deployment | ⭐⭐⭐⭐⭐ | ❌ | ✅ Best |
| Framework integration | ⭐⭐⭐⭐⭐ | ❌ None | ✅ Extensive |
| Async performance | ⭐⭐⭐⭐⭐ | ❌ No | ✅ Yes (3x) |
| Financial benchmarks | ⭐⭐⭐⭐ | ✅ Built-in | ⚠️ Configurable |
| Minimal dependencies | ⭐⭐⭐ | ✅ Simpler | ⚠️ More deps |
| Multi-provider LLM | ⭐⭐⭐⭐ | ⚠️ 3 providers | ✅ 100+ |
| Enterprise monitoring | ⭐⭐⭐⭐ | ❌ Basic | ✅ Opik |
| Token optimization | ⭐⭐⭐⭐ | ⚠️ Manual | ✅ TOON format |
| Quick prototyping | ⭐⭐⭐⭐ | ⚠️ Setup needed | ✅ Simple API |

---

## 9. Recommendation for OpenEvolve

### Primary Strategy: Use Kayba ACE

**Reasons:**
1. **Production-ready** for Knowledge Engine integration
2. **CrewAI integration** already demonstrated in bridge
3. **Async learning** for performance
4. **Opik observability** for monitoring
5. **Framework extensibility** for future needs

### Secondary Strategy: Keep Custom ACE

**Reasons:**
1. **Research validation** - verify Kayba ACE results
2. **Benchmarking** - paper reproduction for academic work
3. **Financial tasks** - pre-configured processors
4. **Fallback option** - if Kayba ACE diverges from paper

### Implementation Pattern

```python
# Production: Use Kayba ACE
from ace import Skillbook, OfflineACE  # Kayba

ace = OfflineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    async_learning=True  # Performance boost
)

# Research: Validate with Custom ACE
from ace import ACE as CustomACE  # Custom

custom_ace = CustomACE(
    api_provider="sambanova",
    generator_model="DeepSeek-V3.1"
)

# Compare results
assert results_close(kayba_results, custom_results)
```

---

## 10. Summary

| Aspect | Custom ACE | Kayba ACE |
|--------|-----------|-----------|
| **Identity** | Research implementation | Production framework |
| **Same Paper?** | ✅ Yes | ✅ Yes |
| **Same Code?** | ❌ No | ❌ No |
| **Best For** | Academia, benchmarking | Production, integration |
| **Learning** | Sequential | Async (3x faster) |
| **Monitoring** | Basic logging | Opik enterprise |
| **Integrations** | Standalone | LangChain, browser-use, CrewAI |
| **Token Format** | Standard | TOON (16-62% savings) |
| **LLM Support** | 3 providers | 100+ providers |

**Bottom Line:** Two independent implementations of the same research, serving different purposes. Use **Kayba ACE for production** and **Custom ACE for research validation**.

---

**Document Version:** 1.0
**Last Updated:** 2025-02-03

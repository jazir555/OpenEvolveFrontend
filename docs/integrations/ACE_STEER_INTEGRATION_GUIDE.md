# ACE + Steer Unified "Active Reliability" Integration Guide

**Document Version:** 1.0
**Date:** January 1, 2026
**Project:** OpenEvolve Frontend - Sovereign-Grade Decomposition Workflow
**Integration Status:** ✅ **FULLY INTEGRATED**

---

## 1. Executive Summary

### 1.1 Purpose
The **ACE + Steer Integration** establishes a unified "Active Reliability" layer across all OpenEvolve agents and workflows. This system combines the self-improving capabilities of the **Agentic Context Engine (ACE)** with the deterministic verification power of **Steer (Reality Locks)**.

The result is a **closed-loop system** where every action is verified against deterministic rules, and every failure is immediately converted into a learning opportunity to prevent future mistakes.

### 1.2 The "Active Reliability" Loop
1.  **Inject (ACE):** Enhanced prompts with model-specific skills retrieved from JSON skillbooks.
2.  **Execute (LLM):** Perform the task using the enhanced context.
3.  **Verify (Steer):** Pass the output through "Reality Locks" (JsonJudge, SlopJudge, etc.).
4.  **Learn (ACE):** If verification fails, Steer extracts the rationale/fix and triggers ACE to learn and update the skillbook.

---

## 2. Technical Architecture

### 2.1 The Bridge (`ace_steer_integration.py`)
The `AceSteerBridge` class is the central orchestrator. It manages the coordination between ACE components and Steer judges.

```python
class AceSteerBridge:
    def __init__(self, ace_agent_id, skillbook_path):
        # Initializes ACE SkillManager, Reflector, and Steer judges
        ...

    def prepare_prompt(self, task, model):
        # 1. ACE Skill Injection
        ...

    def verify_and_learn(self, query, output, verifications, model):
        # 2. Steer Verification
        # 3. ACE Learning from failures
        ...
```

### 2.2 Global Integration Layer (`llm_utils.py`)
To ensure total system coverage, the bridge is integrated into the foundational `llm_utils.py`. Since almost all OpenEvolve modules use `llm_utils._request_openai_compatible_chat`, they automatically inherit Active Reliability.

---

## 3. Integrated Systems

### 3.1 Workflow Engine (Stages 0-4)
The core Sovereign solver now uses the bridge to ensure that problem analysis, decomposition, solving, and reassembly are all verified and improved.
- **File:** `workflow_engine.py`

### 3.2 MAKER & MDAP (Zero-Error Execution)
The bridge enhances the MAKER "first-to-ahead-by-k" voting mechanism by ensuring each individual vote is checked for "slop" and structural integrity before being counted.
- **Files:** `mdap_maker_complete.py`, `openevolve_maker_integration.py`, `maker_engine.py`, `mdap_engine.py`

### 3.3 ROMA (Recursive Decomposition)
Recursive Meta-Agents now use the bridge at every level of the hierarchy, ensuring that parent-task decomposition and leaf-task execution are both reliable.
- **Files:** `roma_mcp_tools.py`, `roma_mdap_maker_engine.py`

### 3.4 LeanAide (Formal Mathematics)
Formal proof generation and autoformalization are verified using Steer's structure checks, with ACE learning from translation errors.
- **Files:** `leanaide_mdap.py`, `leanaide_autoformalization_mdap_maker.py`

### 3.5 Tripartite Teams (Red/Blue/Evaluator)
- **Red Team:** Uses Steer to ensure critiques are actionable and well-formatted.
- **Blue Team:** ACE learns from failed fixes to improve implementation strategies.
- **Evaluator Team:** Deterministic verification of assessment scores.
- **Files:** `blue_team.py`, `red_team.py`, `evaluator_team.py`

---

## 4. Usage Guide

### 4.1 Manual Integration
For new components, use the following pattern:

```python
from ace_steer_integration import AceSteerBridge

# Initialize
bridge = AceSteerBridge(ace_agent_id="my_unique_agent", skillbook_path="./skills.json")

# 1. Prepare (Inject Skills)
enhanced_task = bridge.prepare_prompt(task="My Task", model="gpt-4o")

# 2. Execute (LLM Call)
response = call_llm(enhanced_task)

# 3. Verify & Learn
result = bridge.verify_and_learn(
    query="My Task",
    output=response,
    verifications=["json", "slop"],
    model="gpt-4o"
)

if not result["all_passed"]:
    print(f"Mistake detected: {result['failed_verifications']}")
    # The bridge has already triggered ACE to learn from this!
```

### 4.2 Configuration Parameters
- `ace_agent_id`: String identifying the agent (used for skill categorization).
- `skillbook_path`: Path to the JSON file where knowledge is persisted.
- `verifications`: List of checks to run (`json`, `slop`, `pii`, `markdown`).

---

## 5. Persistence & Recovery

### 5.1 Skillbooks
Skills are stored in standard JSON format (compatible with ACE TOON). 
- **Default path:** `./ace_skillbook.json`
- **Backup path:** `./ace_checkpoints/`

### 5.2 CrewAI Synchronization
ACE state is synchronized with CrewAI tickets via `crewai_integration.py`. This ensures that if a task is delegated or resumed later, the agent retains its identity and learned skills.

---

## 6. Verification Status

| System | Integration | Status |
|--------|-------------|--------|
| LLM Foundation | `llm_utils.py` | ✅ Operational |
| Multi-Agent | MDAP/MAKER | ✅ Operational |
| Recursive | ROMA | ✅ Operational |
| Math | LeanAide | ✅ Operational |
| Teams | Red/Blue/Gold | ✅ Operational |
| UI | BubbleLabs | ✅ Operational |

**Active Reliability is now standard across the entire OpenEvolve ecosystem.**

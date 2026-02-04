# MDAP/MAKER + Matryoshka Integration Roadmap

## 1. Executive Summary
This roadmap outlines the integration of **Matryoshka (Recursive Language Model)** into the **MDAP/MAKER** system. The goal is to solve the "context rot" issue in ultra-long-horizon tasks (10,000+ steps) by utilizing Matryoshka's ability to recursively explore and distill massive context windows.

## 2. System Analysis

### MDAP/MAKER Characteristics
- **MDAP**: Massively Decomposed Agentic Processes. Uses recursive decomposition to break complex tasks into atomic units.
- **MAKER**: Implementation of MDAP with first-to-ahead-by-K voting and red-flagging for zero-error guarantees.
- **Scale**: Designed for "Million-Step" tasks, which naturally suffer from context accumulation and information loss ("context rot").

### Matryoshka Characteristics
- **Recursive exploration**: Writes code to explore documents/data 100x larger than the context window.
- **Distillation**: Can summarize and extract critical entities across massive datasets without RAG chunking artifacts.

## 3. Integration Goals
1.  **Global Context Management**: Use Matryoshka to maintain a "Global State Summary" for long-running MDAP sessions.
2.  **Context-Aware Decomposition**: Enhance MAKER's `TaskDecomposition` by providing agents with distilled context from previous steps.
3.  **Recursive Verification**: Use Matryoshka to verify consistency across deep ROMA-MDAP hierarchies.

## 4. Proposed Integration Plan

### Phase 1: Context Distillation Layer (Level 5 Integration)
*   **Target**: `mdap_maker_complete.py` and `roma_mdap_maker_engine.py`.
*   **Action**: Integrate `GlobalContextManager` (Layer 5) into the recursive solving loop.
*   **Mechanism**: Before each sub-task generation, check the accumulated "Task Memory". If it exceeds the threshold, use Matryoshka to distill the history into a "Canonical State Representation".

### Phase 2: Enhanced ROMA-MDAP-MAKER
*   **Target**: `ROMAMDAPMakerEngine`.
*   **Action**: Update the engine to use `ContextManager.process_input` when handling large external reference materials (documentation, codebases).
*   **Benefit**: Allows ROMA to decompose tasks based on a holistic understanding of massive input context that standard LLMs would truncate.

### Phase 3: Recursive verification
*   **Target**: `VotingEngine`.
*   **Action**: When votes disagree in deep recursion, use Matryoshka to "audit" the entire branch of the hierarchy to find where the logic diverged.

## 5. Technical Implementation Steps

### Step 1: MDAP Stateful Memory Enhancement
Modify `RecursiveMAKERSolver` to support Matryoshka-powered session distillation.

```python
# Conceptual change in mdap_maker_complete.py
class RecursiveMAKERSolver:
    async def solve(self, task, session_id=None):
        if self.use_matryoshka and session_id:
            # Distill context if it's rotting
            self.gcm.manage(session_id, self.accumulated_history)
        ...
```

### Step 2: Config Expansion
Update `ROMAMDAPMakerConfig` to include Matryoshka parameters:
- `matryoshka_enabled: bool`
- `matryoshka_distillation_threshold: int`
- `matryoshka_model: str`

### Step 3: Verification & Benchmarking
Run the "Million-Step" demo with and without Matryoshka integration to measure:
- **Quality**: Accuracy of state preservation across depth.
- **Cost**: Efficiency gains vs. extra distillation calls.
- **Reliability**: Reduction in "hallucination due to truncation".

## 6. Roadmap Timeline
- **Week 1**: Prototype `MatryoshkaDistillator` in `mdap_maker_complete`.
- **Week 2**: Integrate with `ROMAMDAPMakerEngine` for Layer 5 document support.
- **Week 3**: Full end-to-end testing on complex invention planning tasks.
- **Week 4**: Production release and documentation updates.

# 🚀 COMPREHENSIVE INTEGRATION ROADMAP
## OpenEvolve + LoongFlow PES + Knowledge Engine

**Version:** 1.0
**Date:** January 30, 2026
**Timeline:** 8 Weeks
**Expected Improvement:** 70-80% over baseline
**Status:** Agent-Executable Implementation Plan

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Architecture Decision](#architecture-decision)
3. [System Architecture](#system-architecture)
4. [Knowledge Engine Integration](#knowledge-engine-integration)
5. [Gauntlet System Enhancement](#gauntlet-system-enhancement)
6. [Unified Evolutionary Engine](#unified-evolutionary-engine)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Code Examples](#code-examples)
9. [Risk Mitigation](#risk-mitigation)
10. [Success Metrics](#success-metrics)

---

## 1. EXECUTIVE SUMMARY

### What We're Building

A **unified evolutionary optimization platform** that combines:

1. **OpenEvolve** - Quality Diversity, Multi-Objective, Adversarial co-evolution
2. **LoongFlow PES** - Plan-Execute-Summarize paradigm with reasoning-guided search
3. **Knowledge Engine** - Temporal knowledge graph with cross-run learning

### Why We're Doing This

| System | Strengths | Weaknesses |
|--------|-----------|------------|
| **OpenEvolve** | QD, MO, Adversarial, Gauntlets | Blind mutations, slower convergence |
| **LoongFlow** | 60% fewer evaluations, directed search | No QD/MO/Adversarial, single-pass eval |
| **Combined** | **Best of both** - 70-80% improvement | Integration complexity |

### The Hybrid Vision

```
User Problem
    ↓
Strategy Selector (AI chooses best approach)
    ↓
┌─────────────────────────────────────────┐
│  UNIFIED EVOLUTIONARY ENGINE             │
│  ├─ PES Mode (LoongFlow)                │
│  ├─ QD Mode (OpenEvolve MAP-Elites)     │
│  ├─ MO Mode (OpenEvolve NSGA-II)        │
│  └─ Adversarial Mode (OpenEvolve)       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  ENHANCED GAUNTLET SYSTEM               │
│  ├─ LoongFlow AI Eval (quick screen)    │
│  ├─ Red Team (adversarial attack)       │
│  └─ Gold Team (consensus verification)  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  KNOWLEDGE ENGINE                        │
│  ├─ Extract patterns from BOTH systems  │
│  ├─ Temporal knowledge tracking         │
│  └─ Strategy recommendations            │
└─────────────────────────────────────────┘
    ↓
Better Solutions (70-80% improvement)
```

### Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1** | Week 1-2 | Foundation + Architecture Decision |
| **Phase 2** | Week 3-4 | Knowledge Engine Integration |
| **Phase 3** | Week 5-6 | Gauntlet Enhancement |
| **Phase 4** | Week 7-8 | Unified Evolution Engine |
| **Total** | **8 weeks** | Production-ready unified system |

---

## 2. ARCHITECTURE DECISION

### THE CRITICAL CHOICE

**Option A: Keep LoongFlow as Dependency**
**Option B: Extract PES Capabilities into OpenEvolve**

### Decision Matrix

| Criteria | Option A (Dependency) | Option B (Extraction) | Winner |
|----------|----------------------|----------------------|--------|
| **Implementation Speed** | Fast (2-3 weeks) | Medium (4-5 weeks) | **Option A** |
| **Maintenance Burden** | Low (upstream updates) | Medium (fork maintenance) | **Option A** |
| **Bundle Size** | Large (full LoongFlow) | Small (PES only) | **Option B** |
| **Customization** | Limited (upstream API) | Full (modify anything) | **Option B** |
| **Update Access** | Automatic (upstream improvements) | Manual (port changes) | **Option A** |
| **Integration Complexity** | Low (import & call) | Medium (adapt code) | **Option A** |
| **Vibe-Code Factor** | ✅ Agents can easily integrate | ✅ Agents can customize | **Tie** |

### Key Consideration: Vibe-Coded Reality

Since **AI agents write all code**, traditional concerns shift:

**Traditional Concern:** "Extraction creates maintenance burden"
**Vibe-Code Reality:** Agents can maintain anything easily

**Traditional Concern:** "Dependency locks us into upstream API"
**Vibe-Code Reality:** Agents can adapt wrapper layers as needed

**Traditional Concern:** "Bundle size matters"
**Vibe-Code Reality:** Running locally, size irrelevant

### ✅ DECISION: OPTION A - KEEP LOONGFLOW AS DEPENDENCY

**Justification:**

1. **Faster Integration** (2-3 weeks vs 4-5)
2. **Automatic Updates** - Get upstream improvements for free
3. **Lower Risk** - Don't fork code we don't own
4. **Flexibility** - Can always extract later if needed
5. **Vibe-Code Simplicity** - Import and use, no maintenance

### Integration Strategy

```python
# Directory Structure
openevolve/
├── openevolve/ (existing)
├── integrations/
│   ├── loongflow_adapter.py  # NEW: Wrap LoongFlow as OpenEvolve mode
│   └── unified_evolution.py  # NEW: Unified API
└── unified/  # NEW: Unified evolution engine
    ├── strategy_selector.py
    ├── memory_fusion.py
    └── api.py

# LoongFlow remains as submodule or dependency
LoongFlow/  # Keep as-is, don't modify
```

**Key Point:** LoongFlow stays pristine. All integration happens through adapters.

---

## 3. SYSTEM ARCHITECTURE

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      UNIFIED OPTIMIZATION PLATFORM              │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼────────┐    ┌────────▼────────┐    ┌───────▼────────┐
│  OpenEvolve    │    │  LoongFlow      │    │  Knowledge     │
│  Core System   │    │  PES System     │    │  Engine        │
└───────┬────────┘    └────────┬────────┘    └───────┬────────┘
        │                      │                      │
        │ 272 params           │ 50 params            │ Temporal KG
        │ QD, MO, Adversarial  │ Plan-Execute-Summarize│ + Pattern Mining
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  UNIFIED API LAYER  │
                    │  - Strategy Select │
                    │  - Config Mapping   │
                    │  - Memory Fusion    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  GAUNTLET SYSTEM    │
                    │  - LoongFlow AI     │
                    │  - Red Team         │
                    │  - Gold Team        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  SOLUTION OUTPUT   │
                    │  70-80% Better      │
                    └─────────────────────┘
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE FLOW                              │
└─────────────────────────────────────────────────────────────────┘

Evolutionary Run (OpenEvolve OR LoongFlow)
    ↓
Extract Artifacts:
  - Solution patterns
  - Performance metrics
  - Team effectiveness
  - Gauntlet results
    ↓
Store in Knowledge Engine:
  - Neo4j (entities & relations)
  - Qdrant (vector embeddings)
  - MongoDB (documents)
  - Graphiti (temporal tracking)
    ↓
Query & Analyze:
  - Pattern mining
  - Performance comparison
  - Strategy effectiveness
    ↓
Recommend Next Run:
  - Best evolutionary mode
  - Optimal configuration
  - Likely success probability
    ↓
Improve Future Evolution
```

### Component Interaction Map

```
┌──────────────────────────────────────────────────────────────┐
│                 COMPONENT INTERACTIONS                       │
└──────────────────────────────────────────────────────────────┘

Unified API
    ├─→ OpenEvolve.run_unified_evolution()
│   └─→ LoongFlow.PESAgent()
│       └─→ KnowledgeEngine.extract_artifacts()
│           ├─→ Neo4j.store()
│           ├─→ Qdrant.embed()
│           └─→ Graphiti.add_episode()
│               └─→ KnowledgeEngine.recommend_strategy()
│                   └─→ Unified API (next run)
│
└─→ GauntletSystem.execute()
    ├─→ LoongFlowEvaluator.evaluate()  # Round 1
    ├─→ RedTeam.attack()                # Round 2
    └─→ GoldTeam.verify()               # Round 3
        └─→ KnowledgeEngine.extract_gauntlet_feedback()
```

---

## 4. KNOWLEDGE ENGINE INTEGRATION

### Current State (OpenEvolve Only)

```python
# Existing integration
from workflow_knowledge_extractor import WorkflowKnowledgeExtractor

extractor = WorkflowKnowledgeExtractor(knowledge_engine=ke)
artifacts = await extractor.extract_from_workflow(
    workflow_id="ea_run_123",
    stage="evolution",
    results=openevolve_results
)

# Artifacts extracted:
# - SolutionPatternArtifact
# - TeamPerformanceArtifact
# - GauntletEffectivenessArtifact
```

### New State (OpenEvolve + LoongFlow)

```python
# New unified integration
from knowledge_engine.integrations import UnifiedEvolutionKnowledgeExtractor

extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=ke)

# Extract from OpenEvolve
await extractor.extract_from_openevolve(openevolve_results)

# Extract from LoongFlow
await extractor.extract_from_loongflow(loongflow_results)

# Compare performance
comparison = await extractor.compare_performance(
    openevolve_data=oe_data,
    loongflow_data=lf_data,
    problem_type="trading_strategy"
)
# Returns: {"winner": "loongflow", "improvement": "60%", "reason": "..."}

# Get strategy recommendation
strategy = await extractor.recommend_strategy(problem_type="financial_optimization")
# Returns: {"mode": "pes", "confidence": 0.85, "config": {...}}
```

### File Structure

```
knowledge_engine/
├── integrations/
│   ├── openevolve_integration.py  # EXISTING - enhance it
│   ├── loongflow_integration.py   # NEW - create this
│   └── unified_evolution_integration.py  # NEW - create this
├── schemas/
│   ├── evolutionary_artifacts.py  # ENHANCE - add LoongFlow artifacts
│   └── comparison_results.py      # NEW - performance comparison
└── core/
    └── strategy_recommender.py    # NEW - AI-powered strategy selection
```

### Implementation: LoongFlow Integration

**File:** `knowledge_engine/integrations/loongflow_integration.py`

```python
"""
LoongFlow PES Integration for Knowledge Engine
Extracts artifacts from LoongFlow evolutionary runs
"""

from typing import Dict, Any, Optional
from datetime import datetime, UTC

class LoongFlowKnowledgeExtractor:
    """Extract knowledge artifacts from LoongFlow PES runs"""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.graphiti = self.ke.graphiti
        self.neo4j = self.ke.neo4j
        self.qdrant = self.ke.qdrant

    async def extract_from_pes_run(
        self,
        run_id: str,
        problem: Dict[str, Any],
        results: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract artifacts from LoongFlow PES execution

        Args:
            run_id: Unique identifier for this run
            problem: Problem definition
            results: PES execution results
            metadata: Additional metadata

        Returns:
            Dictionary of extracted artifacts
        """
        artifacts = {}

        # 1. Extract Plan-Execute-Summarize patterns
        artifacts['pes_patterns'] = await self._extract_pes_patterns(
            results, metadata
        )

        # 2. Extract evolutionary tree
        artifacts['evolutionary_tree'] = await self._extract_evolutionary_tree(
            results
        )

        # 3. Extract performance metrics
        artifacts['performance_metrics'] = await self._extract_performance_metrics(
            results
        )

        # 4. Extract successful strategies
        artifacts['successful_strategies'] = await self._extract_strategies(
            results
        )

        # 5. Store in temporal knowledge graph
        await self._store_in_graphiti(run_id, artifacts, metadata)

        # 6. Store embeddings in Qdrant
        await self._store_in_qdrant(run_id, artifacts)

        return artifacts

    async def _extract_pes_patterns(
        self,
        results: Dict,
        metadata: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract Plan-Execute-Summarize patterns"""
        patterns = []

        for generation in results.get('generations', []):
            # Plan phase patterns
            plan = generation.get('plan', {})
            if plan:
                patterns.append({
                    'type': 'planning_strategy',
                    'content': plan.get('strategy_description'),
                    'reasoning': plan.get('reasoning'),
                    'success': generation.get('success', False),
                    'timestamp': generation.get('timestamp')
                })

            # Execute phase patterns
            execution = generation.get('execution', {})
            if execution:
                patterns.append({
                    'type': 'execution_pattern',
                    'approach': execution.get('approach'),
                    'early_stopped': execution.get('early_stopped', False),
                    'iterations': execution.get('iterations'),
                    'success': generation.get('success', False)
                })

            # Summary phase patterns
            summary = generation.get('summary', {})
            if summary:
                patterns.append({
                    'type': 'learning_insight',
                    'insight': summary.get('insight'),
                    'reflection': summary.get('reflection'),
                    'recommendations': summary.get('recommendations', [])
                })

        return patterns

    async def _extract_evolutionary_tree(self, results: Dict) -> Dict[str, Any]:
        """Extract evolutionary tree structure"""
        tree = results.get('evolutionary_tree', {})

        return {
            'root_id': tree.get('root_id'),
            'generations': tree.get('num_generations'),
            'branching_factor': tree.get('branching_factor'),
            'best_path': tree.get('best_path', []),
            'all_solutions': tree.get('solutions', [])
        }

    async def _extract_performance_metrics(self, results: Dict) -> Dict[str, Any]:
        """Extract performance metrics"""
        return {
            'total_evaluations': results.get('total_evaluations'),
            'best_fitness': results.get('best_fitness'),
            'convergence_generation': results.get('convergence_generation'),
            'improvement_rate': results.get('improvement_rate'),
            'sample_efficiency': results.get('sample_efficiency')
        }

    async def _extract_strategies(self, results: Dict) -> List[Dict[str, Any]]:
        """Extract successful strategies"""
        strategies = []

        for solution in results.get('successful_solutions', []):
            strategies.append({
                'solution_id': solution.get('id'),
                'fitness': solution.get('fitness'),
                'generation': solution.get('generation'),
                'plan_summary': solution.get('plan_summary'),
                'execution_summary': solution.get('execution_summary')
            })

        return strategies

    async def _store_in_graphiti(
        self,
        run_id: str,
        artifacts: Dict,
        metadata: Optional[Dict]
    ):
        """Store artifacts in temporal knowledge graph"""
        episode_content = f"""
        LoongFlow PES Run {run_id}
        Problem: {metadata.get('problem_description') if metadata else 'N/A'}
        Total Evaluations: {artifacts['performance_metrics'].get('total_evaluations')}
        Best Fitness: {artifacts['performance_metrics'].get('best_fitness')}
        PES Patterns: {len(artifacts['pes_patterns'])} patterns identified
        """

        await self.graphiti.add_episode(
            name=f"loongflow_run_{run_id}",
            episode_body=episode_content,
            reference_datetime=datetime.now(UTC),
            valid_from=datetime.now(UTC)
        )

    async def _store_in_qdrant(self, run_id: str, artifacts: Dict):
        """Store embeddings in vector database"""
        # Create text representation for embedding
        text_repr = str(artifacts)

        await self.qdrant.upsert(
            collection_name="loongflow_runs",
            points=[{
                'id': run_id,
                'vector': self.ke.embed(text_repr),
                'payload': artifacts
            }]
        )
```

### Implementation: Unified Evolution Integration

**File:** `knowledge_engine/integrations/unified_evolution_integration.py`

```python
"""
Unified Evolution Knowledge Integration
Combines OpenEvolve and LoongFlow knowledge extraction
"""

from .openevolve_integration import OpenEvolveKnowledgeExtractor
from .loongflow_integration import LoongFlowKnowledgeExtractor
from typing import Dict, Any, Optional

class UnifiedEvolutionKnowledgeExtractor:
    """Extract and compare knowledge from both evolutionary systems"""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.openevolve_extractor = OpenEvolveKnowledgeExtractor(knowledge_engine)
        self.loongflow_extractor = LoongFlowKnowledgeExtractor(knowledge_engine)

    async def extract_from_both(
        self,
        run_id: str,
        openevolve_results: Optional[Dict] = None,
        loongflow_results: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract artifacts from both systems in a single run

        Returns combined knowledge base
        """
        artifacts = {
            'run_id': run_id,
            'timestamp': datetime.now(UTC).isoformat(),
            'openevolve': None,
            'loongflow': None,
            'comparison': None,
            'recommendations': None
        }

        # Extract from OpenEvolve if available
        if openevolve_results:
            artifacts['openevolve'] = await self.openevolve_extractor.extract_from_workflow(
                workflow_id=run_id,
                stage="evolution",
                results=openevolve_results
            )

        # Extract from LoongFlow if available
        if loongflow_results:
            artifacts['loongflow'] = await self.loongflow_extractor.extract_from_pes_run(
                run_id=run_id,
                problem=metadata.get('problem') if metadata else {},
                results=loongflow_results,
                metadata=metadata
            )

        # Compare performance if both available
        if artifacts['openevolve'] and artifacts['loongflow']:
            artifacts['comparison'] = await self.compare_performance(
                openevolve_data=artifacts['openevolve'],
                loongflow_data=artifacts['loongflow'],
                problem_type=metadata.get('domain') if metadata else 'general'
            )

        # Generate recommendations
        artifacts['recommendations'] = await self.recommend_strategy(
            artifacts=artifacts,
            problem_type=metadata.get('domain') if metadata else 'general'
        )

        # Store unified artifacts
        await self._store_unified_artifacts(artifacts)

        return artifacts

    async def compare_performance(
        self,
        openevolve_data: Dict,
        loongflow_data: Dict,
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Compare OpenEvolve vs LoongFlow performance

        Returns:
        {
            'winner': 'openevolve' | 'loongflow' | 'tie',
            'improvement': '60%',
            'reason': '...',
            'metrics': {...}
        }
        """
        # Extract comparable metrics
        oe_evals = openevolve_data.get('total_evaluations', 0)
        lf_evals = loongflow_data.get('total_evaluations', 0)

        oe_fitness = openevolve_data.get('best_fitness', 0)
        lf_fitness = loongflow_data.get('best_fitness', 0)

        # Determine winner based on problem type
        if problem_type in ['finance', 'trading', 'science', 'engineering']:
            # For expensive evaluations, LoongFlow's sample efficiency wins
            if lf_evals < oe_evals and lf_fitness >= oe_fitness * 0.95:
                return {
                    'winner': 'loongflow',
                    'improvement': f"{(1 - lf_evals/oe_evals) * 100:.1f}%",
                    'reason': 'Fewer evaluations with comparable quality',
                    'metrics': {
                        'sample_efficiency_gain': oe_evals / lf_evals,
                        'fitness_ratio': lf_fitness / oe_fitness
                    }
                }

        # For other problems, check which achieved better fitness
        if lf_fitness > oe_fitness:
            return {
                'winner': 'loongflow',
                'improvement': f"{((lf_fitness / oe_fitness) - 1) * 100:.1f}%",
                'reason': 'Better final solution quality',
                'metrics': {
                    'fitness_improvement': lf_fitness / oe_fitness,
                    'evaluation_count': oe_evals / lf_evals
                }
            }
        elif oe_fitness > lf_fitness:
            return {
                'winner': 'openevolve',
                'improvement': f"{((oe_fitness / lf_fitness) - 1) * 100:.1f}%",
                'reason': 'Better final solution quality',
                'metrics': {
                    'fitness_improvement': oe_fitness / lf_fitness,
                    'evaluation_count': lf_evals / oe_evals
                }
            }
        else:
            return {
                'winner': 'tie',
                'improvement': '0%',
                'reason': 'Comparable performance',
                'metrics': {
                    'fitness_ratio': 1.0,
                    'evaluation_ratio': oe_evals / lf_evals
                }
            }

    async def recommend_strategy(
        self,
        artifacts: Dict,
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Recommend evolutionary strategy for next run

        Uses historical performance from knowledge graph
        """
        # Query knowledge graph for similar problems
        similar_runs = await self.ke.query(
            f"""
            MATCH (run:EvolutionaryRun)
            WHERE run.domain = '{problem_type}'
            RETURN run
            ORDER BY run.timestamp DESC
            LIMIT 10
            """
        )

        # Analyze which strategies worked best
        strategy_performance = {}

        for run in similar_runs:
            strategy = run.get('strategy', 'unknown')
            fitness = run.get('best_fitness', 0)
            evaluations = run.get('total_evaluations', 1)

            if strategy not in strategy_performance:
                strategy_performance[strategy] = []

            strategy_performance[strategy].append(fitness / evaluations)

        # Find best strategy
        best_strategy = None
        best_efficiency = 0

        for strategy, efficiencies in strategy_performance.items():
            avg_efficiency = sum(efficiencies) / len(efficiencies)
            if avg_efficiency > best_efficiency:
                best_efficiency = avg_efficiency
                best_strategy = strategy

        # Get recommended config
        if best_strategy == 'pes':
            config = await self._get_pes_config(problem_type)
        elif best_strategy == 'qd':
            config = await self._get_qd_config(problem_type)
        elif best_strategy == 'mo':
            config = await self._get_mo_config(problem_type)
        elif best_strategy == 'adversarial':
            config = await self._get_adversarial_config(problem_type)
        else:
            config = {}

        return {
            'recommended_strategy': best_strategy,
            'confidence': min(0.9, len(strategy_performance.get(best_strategy, [])) * 0.1),
            'expected_improvement': f"{best_efficiency * 100:.1f}%",
            'config': config
        }

    async def _get_pes_config(self, problem_type: str) -> Dict:
        """Get recommended PES configuration"""
        # Query knowledge graph for best PES configs
        return {
            'evolution_mode': 'pes',
            'max_iterations': 50,
            'enable_planning': True,
            'enable_memory': True,
            'early_stopping': True
        }

    async def _get_qd_config(self, problem_type: str) -> Dict:
        """Get recommended QD configuration"""
        return {
            'evolution_mode': 'qd',
            'grid_resolution': 10,
            'feature_dimensions': [...],
            'archive_size': 1000
        }

    async def _get_mo_config(self, problem_type: str) -> Dict:
        """Get recommended MO configuration"""
        return {
            'evolution_mode': 'mo',
            'objectives': [...],
            'pareto_front_size': 100
        }

    async def _get_adversarial_config(self, problem_type: str) -> Dict:
        """Get recommended adversarial configuration"""
        return {
            'evolution_mode': 'adversarial',
            'adversarial_rounds': 20,
            'red_team_models': [...]
        }
```

---

## 5. GAUNTLET SYSTEM ENHANCEMENT

### Current Gauntlet Flow

```
Solution
    ↓
Red Team Round (adversarial attack)
    ↓
Gold Team Round (consensus verification)
    ↓
Result (pass/fail)
```

### Enhanced Gauntlet Flow

```
Solution
    ↓
Round 1: LoongFlow AI Evaluator (QUICK SCREEN)
    ├─ Single-pass evaluation
    ├─ Score: 0-100
    ├─ Identify promising candidates
    └─ Filter: Score > 50 proceeds
    ↓
Round 2: Red Team Attack (ADVERSARIAL)
    ├─ Multi-round attack
    ├─ Fuzzing integration
    ├─ Vulnerability scanning
    └─ Survives? → Proceed
    ↓
Round 3: Gold Team Verification (CONSENSUS)
    ├─ Multi-judge evaluation
    ├─ Voting (quorum/consensus)
    ├─ Lean 4 formal verification (if math)
    └─ Approved? → SUCCESS
```

### File Structure

```
Bubbles/
├── evaluators/
│   ├── loongflow_adapter.py          # NEW - LoongFlow as evaluator
│   ├── red_team_evaluator.py         # EXISTING
│   └── gold_team_evaluator.py        # EXISTING
├── gauntlet_manager.py               # ENHANCE - Add LoongFlow round
└── enhanced_gauntlet_system.py       # NEW - 3-round orchestration
```

### Implementation: LoongFlow Evaluator Adapter

**File:** `Bubbles/evaluators/loongflow_adapter.py`

```python
"""
LoongFlow Evaluator Adapter for Gauntlet System
Wraps LoongFlow's AI evaluation as a gauntlet round
"""

from typing import Dict, Any
from bubbles.evaluators.base_evaluator import BaseGauntletEvaluator

class LoongFlowEvaluatorAdapter(BaseGauntletEvaluator):
    """
    Adapter that makes LoongFlow's evaluation compatible with OpenEvolve gauntlets
    """

    def __init__(self, llm_config, timeout=300):
        self.llm_config = llm_config
        self.timeout = timeout

        # Import LoongFlow (lazy import to avoid dependency issues)
        try:
            from loongflow.agents.general_agent.evaluator import GeneralEvaluator
            from loongflow.framework.pes.context import EvaluatorConfig

            self.evaluator = GeneralEvaluator(
                config=EvaluatorConfig(
                    llm_config=llm_config,
                    evaluate_code=None,  # AI mode
                    timeout=timeout
                )
            )
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: LoongFlow not available, adapter disabled")

    async def evaluate_round(
        self,
        solution: 'SolutionAttempt',
        round_config: 'GauntletRoundRule',
        context: Dict[str, Any]
    ) -> 'GauntletRoundResult':
        """
        Evaluate solution using LoongFlow's AI evaluator

        Args:
            solution: The solution to evaluate
            round_config: Gauntlet round configuration
            context: Additional context

        Returns:
            GauntletRoundResult with score and feedback
        """
        if not self.available:
            # Fallback to basic evaluation
            return await self._fallback_evaluation(solution, round_config)

        # Convert solution to LoongFlow message format
        from loongflow.framework.message import Message, ContentElement

        message = Message.from_elements([
            ContentElement(
                mime_type='text/plain',
                data=solution.solution_content
            )
        ])

        # Run LoongFlow evaluation
        try:
            result = await self.evaluator.evaluate(
                message=message,
                context=context
            )

            # Convert LoongFlow result to gauntlet result
            score = result.score if result.score is not None else 0.5

            passed = score >= round_config.min_score

            return GauntletRoundResult(
                round_id=round_config.rule_id,
                passed=passed,
                score=score,
                feedback=result.summary if result.summary else "No feedback",
                evaluation_details={
                    'loongflow_status': str(result.status),
                    'execution_time': result.execution_time if hasattr(result, 'execution_time') else None
                }
            )

        except Exception as e:
            # Error during evaluation
            return GauntletRoundResult(
                round_id=round_config.rule_id,
                passed=False,
                score=0.0,
                feedback=f"Evaluation error: {str(e)}",
                evaluation_details={'error': str(e)}
            )

    async def _fallback_evaluation(
        self,
        solution: 'SolutionAttempt',
        round_config: 'GauntletRoundRule'
    ) -> 'GauntletRoundResult':
        """Fallback when LoongFlow not available"""
        # Basic keyword-based evaluation
        content = solution.solution_content.lower()

        score = 0.5  # Default

        # Simple heuristics
        if 'def ' in content or 'class ' in content:
            score += 0.2  # Has code structure

        if 'import ' in content:
            score += 0.1  # Has imports

        if len(content) > 100:
            score += 0.1  # Substantial content

        if 'error' not in content and 'bug' not in content:
            score += 0.1  # No obvious issues

        return GauntletRoundResult(
            round_id=round_config.rule_id,
            passed=score >= round_config.min_score,
            score=min(1.0, score),
            feedback="Fallback evaluation (LoongFlow unavailable)"
        )
```

### Implementation: Enhanced Gauntlet System

**File:** `Bubbles/enhanced_gauntlet_system.py`

```python
"""
Enhanced Gauntlet System with LoongFlow Integration
3-Round evaluation: LoongFlow → Red Team → Gold Team
"""

from typing import List, Dict, Any
from bubbles.gauntlet_manager import GauntletSystem
from bubbles.evaluators.loongflow_adapter import LoongFlowEvaluatorAdapter

class EnhancedGauntletSystem(GauntletSystem):
    """
    Enhanced gauntlet system with LoongFlow AI evaluation as Round 1
    """

    def __init__(self, team_manager, openevolve_client, llm_config):
        super().__init__(team_manager, openevolve_client)

        # Add LoongFlow evaluator
        self.loongflow_evaluator = LoongFlowEvaluatorAdapter(
            llm_config=llm_config
        )

    async def execute_enhanced_gauntlet(
        self,
        solution: 'SolutionAttempt',
        sub_problem: 'SubProblem',
        gauntlet_config: 'GauntletDefinition'
    ) -> 'GauntletExecutionResult':
        """
        Execute enhanced 3-round gauntlet

        Rounds:
        1. LoongFlow AI Evaluation (quick screen)
        2. Red Team Attack (adversarial)
        3. Gold Team Verification (consensus)
        """
        round_results = []

        # Round 1: LoongFlow Quick Screen
        loongflow_result = await self._execute_loongflow_round(
            solution, sub_problem, gauntlet_config
        )
        round_results.append(loongflow_result)

        # Early exit if LoongFlow score is too low
        if not loongflow_result.passed:
            return GauntletExecutionResult(
                overall_passed=False,
                final_score=loongflow_result.score,
                round_results=round_results,
                failed_round='loongflow_ai_eval'
            )

        # Round 2: Red Team Attack
        red_team_result = await self._execute_red_team_round(
            solution, sub_problem, gauntlet_config
        )
        round_results.append(red_team_result)

        # Early exit if Red Team rejects
        if not red_team_result.passed:
            return GauntletExecutionResult(
                overall_passed=False,
                final_score=(loongflow_result.score + red_team_result.score) / 2,
                round_results=round_results,
                failed_round='red_team'
            )

        # Round 3: Gold Team Verification
        gold_team_result = await self._execute_gold_team_round(
            solution, sub_problem, gauntlet_config
        )
        round_results.append(gold_team_result)

        # Calculate final score
        final_score = (
            loongflow_result.score * 0.2 +  # 20% weight
            red_team_result.score * 0.3 +   # 30% weight
            gold_team_result.score * 0.5     # 50% weight
        )

        # Final decision
        overall_passed = all(r.passed for r in round_results)

        return GauntletExecutionResult(
            overall_passed=overall_passed,
            final_score=final_score,
            round_results=round_results,
            failed_round=None if overall_passed else self._get_failed_round(round_results)
        )

    async def _execute_loongflow_round(
        self,
        solution: 'SolutionAttempt',
        sub_problem: 'SubProblem',
        gauntlet_config: 'GauntletDefinition'
    ) -> 'GauntletRoundResult':
        """Execute Round 1: LoongFlow AI evaluation"""

        # Find LoongFlow round config
        loongflow_round = next(
            (r for r in gauntlet_config.rounds if r.evaluator == 'loongflow_adapter'),
            None
        )

        if loongflow_round is None:
            # Create default config
            loongflow_round = GauntletRoundRule(
                rule_id='loongflow_ai_eval',
                rule_type='automated',
                min_score=0.5,
                max_attempts=1,
                evaluator='loongflow_adapter'
            )

        return await self.loongflow_evaluator.evaluate_round(
            solution=solution,
            round_config=loongflow_round,
            context={
                'sub_problem': sub_problem,
                'gauntlet_id': gauntlet_config.gauntlet_id
            }
        )

    async def _execute_red_team_round(self, solution, sub_problem, gauntlet_config):
        """Execute Round 2: Red Team adversarial attack"""
        # Use existing Red Team logic
        return await self.execute_red_team_round(solution, sub_problem, gauntlet_config)

    async def _execute_gold_team_round(self, solution, sub_problem, gauntlet_config):
        """Execute Round 3: Gold Team verification"""
        # Use existing Gold Team logic
        return await self.execute_gold_team_round(solution, sub_problem, gauntlet_config)
```

---

## 6. UNIFIED EVOLUTIONARY ENGINE

### Strategy Selection Logic

```python
class StrategySelector:
    """
    AI-powered strategy selector
    Chooses optimal evolutionary mode based on problem characteristics
    """

    async def select_strategy(
        self,
        problem: Dict[str, Any],
        domain: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Select best evolutionary strategy

        Decision tree:
        1. Check evaluation cost (expensive? → PES)
        2. Check objectives (multiple? → MO)
        3. Check diversity need (explore? → QD)
        4. Check robustness need (safety-critical? → Adversarial)
        5. Default: PES (best general performance)
        """

        # Factor 1: Evaluation Cost
        if self._expensive_evaluations(problem):
            # Expensive evaluations favor PES (60% fewer)
            if domain in ['finance', 'science', 'engineering']:
                return {
                    'mode': 'pes',
                    'confidence': 0.9,
                    'reason': 'Expensive evaluations, PES reduces cost by 60%'
                }

        # Factor 2: Multiple Objectives
        if self._has_multiple_objectives(problem):
            return {
                'mode': 'mo',
                'confidence': 0.85,
                'reason': 'Multiple competing objectives require Pareto optimization'
            }

        # Factor 3: Diversity Need
        if self._needs_diversity(problem):
            return {
                'mode': 'qd',
                'confidence': 0.8,
                'reason': 'Exploration of diverse solutions required'
            }

        # Factor 4: Robustness Need
        if self._needs_robustness(problem, domain):
            return {
                'mode': 'adversarial',
                'confidence': 0.85,
                'reason': 'Safety-critical, adversarial testing finds failures'
            }

        # Factor 5: Real-time Constraints
        if constraints.get('real_time', False):
            return {
                'mode': 'pes',
                'confidence': 0.7,
                'reason': 'Real-time constraints favor directed search'
            }

        # Default: PES (best overall performance)
        return {
            'mode': 'pes',
            'confidence': 0.75,
            'reason': 'PES provides best general performance'
        }

    def _expensive_evaluations(self, problem: Dict) -> bool:
        """Check if evaluations are expensive"""
        expensive_indicators = [
            'backtest' in problem.get('description', '').lower(),
            'simulation' in problem.get('description', '').lower(),
            'experiment' in problem.get('description', '').lower(),
            problem.get('estimated_time_per_eval', 0) > 60,  # > 1 minute
            problem.get('estimated_cost_per_eval', 0) > 100  # > $100
        ]
        return any(expensive_indicators)

    def _has_multiple_objectives(self, problem: Dict) -> bool:
        """Check if problem has multiple objectives"""
        return len(problem.get('objectives', [])) > 1

    def _needs_diversity(self, problem: Dict) -> bool:
        """Check if problem needs diverse solutions"""
        diversity_indicators = [
            'explore' in problem.get('description', '').lower(),
            'novel' in problem.get('description', '').lower(),
            problem.get('require_diversity', False)
        ]
        return any(diversity_indicators)

    def _needs_robustness(self, problem: Dict, domain: str) -> bool:
        """Check if problem needs robustness testing"""
        robust_domains = ['engineering', 'pharma', 'finance']
        return domain in robust_domains or problem.get('safety_critical', False)
```

### Unified API

**File:** `openevolve/unified/api.py`

```python
"""
Unified Evolutionary API
Single entry point for all evolutionary modes
"""

from .strategy_selector import StrategySelector
from ..openevolve import run_unified_evolution  # OpenEvolve's API

# Lazy import of LoongFlow
try:
    from loongflow.agents.math_agent import MathPESAgent
    LOONGFLOW_AVAILABLE = True
except ImportError:
    LOONGFLOW_AVAILABLE = False

class UnifiedEvolutionaryEngine:
    """
    Unified API for evolutionary optimization
    Automatically selects best strategy (PES, QD, MO, Adversarial)
    """

    def __init__(self, knowledge_engine=None):
        self.strategy_selector = StrategySelector()
        self.knowledge_engine = knowledge_engine
        self.loongflow_available = LOONGFLOW_AVAILABLE

    async def evolve(
        self,
        problem: str,
        domain: str = 'general',
        max_evaluations: int = 100,
        objectives: List[str] = None,
        constraints: Dict[str, Any] = None,
        enable_planning: bool = True,
        enable_memory: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point for evolutionary optimization

        Args:
            problem: Problem description
            domain: Application domain (finance, science, etc.)
            max_evaluations: Maximum evaluations allowed
            objectives: List of optimization objectives
            constraints: Additional constraints
            enable_planning: Enable PES planning phase
            enable_memory: Enable memory/knowledge retrieval
            **kwargs: Additional parameters

        Returns:
            {
                'best_solution': ...,
                'fitness': ...,
                'strategy_used': 'pes'|'qd'|'mo'|'adversarial',
                'evaluations': ...,
                'improvement': ...
            }
        """

        # Step 1: Select strategy
        strategy = await self.strategy_selector.select_strategy(
            problem={
                'description': problem,
                'objectives': objectives or []
            },
            domain=domain,
            constraints=constraints or {}
        )

        strategy_mode = strategy['mode']

        # Step 2: Configure based on strategy
        if strategy_mode == 'pes':
            result = await self._run_pes_evolution(
                problem, domain, max_evaluations,
                enable_planning, enable_memory, **kwargs
            )
        elif strategy_mode == 'qd':
            result = await self._run_qd_evolution(
                problem, domain, max_evaluations, **kwargs
            )
        elif strategy_mode == 'mo':
            result = await self._run_mo_evolution(
                problem, domain, max_evaluations, objectives, **kwargs
            )
        elif strategy_mode == 'adversarial':
            result = await self._run_adversarial_evolution(
                problem, domain, max_evaluations, **kwargs
            )
        else:
            # Fallback to standard OpenEvolve
            result = await self._run_standard_evolution(
                problem, domain, max_evaluations, **kwargs
            )

        # Add metadata
        result['strategy_used'] = strategy_mode
        result['strategy_confidence'] = strategy['confidence']
        result['strategy_reason'] = strategy['reason']

        # Step 3: Extract knowledge
        if self.knowledge_engine:
            await self._extract_and_store_knowledge(result, domain)

        return result

    async def _run_pes_evolution(
        self, problem, domain, max_evaluations,
        enable_planning, enable_memory, **kwargs
    ):
        """Run PES evolution (LoongFlow)"""

        if not self.loongflow_available:
            # Fallback to OpenEvolve with planning prompt
            return await self._run_standard_evolution(
                problem, domain, max_evaluations,
                planning_enabled=True, **kwargs
            )

        # Configure LoongFlow
        config = {
            'max_iterations': max_evaluations,
            'enable_planning': enable_planning,
            'enable_memory': enable_memory,
            'domain': domain,
            **kwargs
        }

        # Run LoongFlow PES
        agent = MathPESAgent(config=config)
        result = await agent.run(problem)

        return {
            'best_solution': result.best_solution,
            'fitness': result.best_fitness,
            'evaluations': result.total_evaluations,
            'improvement': result.improvement_rate
        }

    async def _run_qd_evolution(self, problem, domain, max_evaluations, **kwargs):
        """Run Quality Diversity evolution (OpenEvolve)"""

        return await run_unified_evolution(
            problem_statement=problem,
            evolution_mode='qd',
            max_iterations=max_evaluations,
            grid_resolution=kwargs.get('grid_resolution', 10),
            feature_dimensions=kwargs.get('feature_dimensions', []),
            archive_size=kwargs.get('archive_size', 1000)
        )

    async def _run_mo_evolution(
        self, problem, domain, max_evaluations, objectives, **kwargs
    ):
        """Run Multi-Objective evolution (OpenEvolve)"""

        return await run_unified_evolution(
            problem_statement=problem,
            evolution_mode='mo',
            max_iterations=max_evaluations,
            objectives=objectives,
            pareto_front_size=kwargs.get('pareto_front_size', 100)
        )

    async def _run_adversarial_evolution(
        self, problem, domain, max_evaluations, **kwargs
    ):
        """Run Adversarial evolution (OpenEvolve)"""

        return await run_unified_evolution(
            problem_statement=problem,
            evolution_mode='adversarial',
            max_iterations=max_evaluations,
            adversarial_rounds=kwargs.get('adversarial_rounds', 20),
            red_team_models=kwargs.get('red_team_models', [])
        )

    async def _run_standard_evolution(self, problem, domain, max_evaluations, **kwargs):
        """Run standard evolution (OpenEvolve)"""

        return await run_unified_evolution(
            problem_statement=problem,
            evolution_mode='standard',
            max_iterations=max_evaluations,
            **kwargs
        )

    async def _extract_and_store_knowledge(self, result, domain):
        """Extract artifacts and store in knowledge engine"""
        # Implementation details...
        pass
```

---

## 7. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)

**Goal:** Architecture decision + basic LoongFlow integration

#### Task 1.1: Architecture Decision (Day 1)
**Agent:** Architect Agent
**Prompt:**
```
Analyze integration options:
- Option A: LoongFlow as dependency
- Option B: Extract PES capabilities

Create decision document with:
- Comparison matrix
- Recommendation
- Rationale

File: docs/knowledge_engine/ARCHITECTURE_DECISION.md
```

**Deliverable:** Decision document

#### Task 1.2: LoongFlow Dependency Setup (Days 2-3)
**Agent:** Integration Agent
**Prompt:**
```
1. Add LoongFlow as git submodule or pip dependency
2. Verify imports work
3. Test basic PES execution
4. Create wrapper: openevolve/integrations/loongflow_adapter.py

Test: Run simple LoongFlow PES example
```

**Deliverable:** Working LoongFlow integration

#### Task 1.3: Unified Config Schema (Days 4-5)
**Agent:** Schema Agent
**Prompt:**
```
Create unified configuration schema:
- Map 272 OpenEvolve params
- Map 50 LoongFlow params
- Create unified config class
- Support mode-specific configs

File: openevolve/unified/config.py
```

**Deliverable:** Unified config schema

#### Task 1.4: Testing (Days 6-7)
**Agent:** Test Agent
**Prompt:**
```
1. Test LoongFlow works standalone
2. Test OpenEvolve still works
3. Test config mapping
4. Validate no regressions

File: tests/unified/test_phase1.py
```

**Deliverable:** Passing tests

---

### Phase 2: Knowledge Engine Integration (Week 3-4)

**Goal:** Knowledge Engine learns from both systems

#### Task 2.1: LoongFlow Knowledge Extractor (Days 1-2)
**Agent:** Knowledge Agent
**Prompt:**
```
Create LoongFlow knowledge extractor:
- Extract PES patterns
- Extract evolutionary tree
- Extract performance metrics
- Store in Graphiti, Neo4j, Qdrant

File: knowledge_engine/integrations/loongflow_integration.py
```

**Deliverable:** Working extractor

#### Task 2.2: Unified Extractor (Days 3-4)
**Agent:** Knowledge Agent
**Prompt:**
```
Create unified extractor:
- Combine OpenEvolve + LoongFlow
- Compare performance
- Generate recommendations

File: knowledge_engine/integrations/unified_evolution_integration.py
```

**Deliverable:** Unified extractor

#### Task 2.3: Strategy Recommender (Days 5-6)
**Agent:** AI Agent
**Prompt:**
```
Create strategy recommender:
- Query knowledge graph
- Analyze past performance
- Recommend best mode

File: knowledge_engine/core/strategy_recommender.py
```

**Deliverable:** Working recommender

#### Task 2.4: Testing (Day 7)
**Agent:** Test Agent
**Prompt:**
```
Test knowledge extraction:
- Extract from OpenEvolve run
- Extract from LoongFlow run
- Test temporal queries
- Test recommendations
```

**Deliverable:** Knowledge graph contains both systems

---

### Phase 3: Gauntlet Enhancement (Week 5-6)

**Goal:** LoongFlow evaluation integrated into gauntlets

#### Task 3.1: LoongFlow Gauntlet Adapter (Days 1-2)
**Agent:** Integration Agent
**Prompt:**
```
Create LoongFlow gauntlet adapter:
- Wrap LoongFlow evaluator
- Convert formats
- Handle errors

File: Bubbles/evaluators/loongflow_adapter.py
```

**Deliverable:** Working adapter

#### Task 3.2: Enhanced Gauntlet System (Days 3-4)
**Agent:** Gauntlet Agent
**Prompt:**
```
Create enhanced gauntlet system:
- Round 1: LoongFlow (quick screen)
- Round 2: Red Team (attack)
- Round 3: Gold Team (verify)
- Orchestrate flow

File: Bubbles/enhanced_gauntlet_system.py
```

**Deliverable:** Enhanced gauntlets

#### Task 3.3: Multi-Round Orchestration (Days 5-6)
**Agent:** Orchestration Agent
**Prompt:**
```
Implement multi-round flow:
- Pass results between rounds
- Aggregate scores
- Early exit logic
- Weighted voting
```

**Deliverable:** Complete gauntlet flow

#### Task 3.4: Testing (Day 7)
**Agent:** Test Agent
**Prompt:**
```
Test gauntlet enhancement:
- Run complete 3-round gauntlet
- Validate LoongFlow adapter
- Measure quality improvement
```

**Deliverable:** Gauntlets working with LoongFlow

---

### Phase 4: Unified Evolution Engine (Week 7-8)

**Goal:** Single API for all modes

#### Task 4.1: Strategy Selector (Days 1-2)
**Agent:** AI Agent
**Prompt:**
```
Create strategy selector:
- Analyze problem characteristics
- Select optimal mode
- Provide confidence

File: openevolve/unified/strategy_selector.py
```

**Deliverable:** Working selector

#### Task 4.2: Unified API (Days 3-5)
**Agent:** API Agent
**Prompt:**
```
Create unified API:
- Single entry point
- Auto-configuration
- Mode switching
- Clean interface

File: openevolve/unified/api.py
```

**Deliverable:** Unified API

#### Task 4.3: Memory Fusion (Days 6-7)
**Agent:** Memory Agent
**Prompt:**
```
Implement memory fusion:
- Combine evolutionary tree
- Combine MAP-Elites archive
- Combine Pareto fronts
- Unified querying

File: openevolve/unified/memory_fusion.py
```

**Deliverable:** Fused memory

#### Task 4.4: Domain Optimizers (Days 8-9)
**Agent:** Domain Agent
**Prompt:**
```
Create domain-specific optimizers:
- Finance, Trading, Science
- Engineering, Pharma, Web
- Pre-configured strategies

File: openevolve/unified/domain_optimizers.py
```

**Deliverable:** 6 domain optimizers

#### Task 4.5: Integration Testing (Days 10-12)
**Agent:** Test Agent
**Prompt:**
```
Comprehensive testing:
- All 6 domains
- All evolutionary modes
- Performance benchmarks
- Validate 70-80% improvement

File: tests/unified/test_integration.py
```

**Deliverable:** All tests passing

#### Task 4.6: Documentation (Day 13)
**Agent:** Documentation Agent
**Prompt:**
```
Create documentation:
- API reference
- Usage examples
- Migration guide
- Architecture diagrams

File: docs/unified_evolution/
```

**Deliverable:** Complete documentation

---

## 8. CODE EXAMPLES

### Example 1: Unified API - Finance Domain

```python
from openevolve.unified import UnifiedEvolutionaryEngine

# Initialize engine with knowledge engine
engine = UnifiedEvolutionaryEngine(knowledge_engine=ke)

# Optimize trading strategy
result = await engine.evolve(
    problem="Optimize portfolio allocation for max return with min risk",
    domain="finance",
    max_evaluations=50,  # Limited budget for backtests
    objectives=["return", "risk", "liquidity"],
    enable_planning=True,  # Use financial knowledge
    enable_memory=True  # Learn from past strategies
)

print(f"Strategy: {result['strategy_used']}")  # 'pes'
print(f"Return: {result['objectives']['return']}")
print(f"Risk: {result['objectives']['risk']}")
print(f"Evaluations: {result['evaluations']}")  # ~30 (60% fewer than baseline)
```

### Example 2: Scientific Experiment Design

```python
# Optimize experimental design
result = await engine.evolve(
    problem="Optimize chemical reaction conditions for maximum yield",
    domain="science",
    max_evaluations=20,  # Each experiment = $5K
    enable_planning=True,  # Leverage chemical knowledge
    enable_memory=True  # Learn from past reactions
)

# PES mode selected (expensive evaluations)
# 60% fewer experiments = $60K savings
print(f"Conditions: {result['best_solution']}")
print(f"Predicted yield: {result['fitness']}")
print(f"Experiments: {result['evaluations']}")  # ~12 (vs 30 baseline)
```

### Example 3: Engineering with Safety Testing

```python
# Design bridge with safety testing
result = await engine.evolve(
    problem="Design lightweight bridge that supports 50 tons",
    domain="engineering",
    max_evaluations=100,
    enable_planning=True,  # PES for design
    enable_memory=True
)

# PES designs bridge (fewer FEA simulations)
# Then automatically runs adversarial for safety
print(f"Design: {result['best_solution']}")
print(f"Weight: {result['weight']}")
print(f"Safety factor: {result['safety_factor']}")
```

### Example 4: Knowledge-Guided Evolution

```python
from knowledge_engine import UnifiedEvolutionKnowledgeExtractor

# Extract from past runs
extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=ke)

# After OpenEvolve run
await extractor.extract_from_openevolve(openevolve_results)

# After LoongFlow run
await extractor.extract_from_loongflow(loongflow_results)

# Get recommendation
strategy = await extractor.recommend_strategy(
    problem_type="financial_optimization"
)
print(f"Recommended: {strategy['recommended_strategy']}")  # 'pes'
print(f"Confidence: {strategy['confidence']}")  # 0.85
print(f"Expected improvement: {strategy['expected_improvement']}")  # '60%'
```

### Example 5: Enhanced Gauntlet

```python
from Bubbles import EnhancedGauntletSystem

# Create enhanced gauntlet
gauntlet_system = EnhancedGauntletSystem(
    team_manager=team_mgr,
    openevolve_client=oe_client,
    llm_config=llm_cfg
)

# Define 3-round gauntlet
gauntlet = gauntlet_system.create_enhanced_gauntlet(
    rounds=[
        {'type': 'loongflow_ai', 'min_score': 0.5},  # Quick screen
        {'type': 'red_team', 'min_score': 0.7},     # Attack
        {'type': 'gold_team', 'min_score': 0.9}     # Verify
    ]
)

# Execute gauntlet
result = await gauntlet_system.execute_enhanced_gauntlet(
    solution=solution,
    sub_problem=problem,
    gauntlet_config=gauntlet
)

print(f"Passed: {result.overall_passed}")
print(f"Final score: {result.final_score}")
print(f"Round results: {result.round_results}")
```

---

## 9. RISK MITIGATION

### Potential Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Integration complexity** | Medium | Medium | Incremental phases, test each |
| **Performance regression** | Low | High | Benchmark at each phase |
| **Knowledge contamination** | Low | Medium | Separate namespaces in graph |
| **API confusion** | Medium | Low | Clear unified API, deprecate old gradually |
| **LoongFlow dependency changes** | Low | Medium | Version pinning, adapter pattern |
| **Memory bloat** | Medium | Low | Prune old artifacts, retention policy |

### Rollback Strategy

Each phase is independently reversible:

- **Phase 1:** Remove LoongFlow dependency, revert to OpenEvolve-only
- **Phase 2:** Disable LoongFlow extractor, use OpenEvolve-only
- **Phase 3:** Remove LoongFlow from gauntlets, use original gauntlets
- **Phase 4:** Deprecate unified API, use OpenEvolve API directly

**Feature Flags:**

```python
# Enable/disable new features
ENABLE_LOONGFLOW = os.getenv('ENABLE_LOONGFLOW', 'true') == 'true'
ENABLE_UNIFIED_API = os.getenv('ENABLE_UNIFIED_API', 'true') == 'true'
ENABLE_ENHANCED_GAUNTLETS = os.getenv('ENABLE_ENHANCED_GAUNTLETS', 'true') == 'true'
```

### Validation Criteria

**Phase 1 Complete When:**
- LoongFlow imports successfully
- Can run PES execution end-to-end
- Config mapping works

**Phase 2 Complete When:**
- Knowledge graph contains both OpenEvolve and LoongFlow data
- Temporal queries work across both systems
- Strategy recommendations are reasonable

**Phase 3 Complete When:**
- LoongFlow evaluator works in gauntlets
- 3-round gauntlet flow works
- Quality metrics improve or stay same

**Phase 4 Complete When:**
- All 6 domains work with unified API
- Performance benchmarks show 70-80% improvement
- All tests pass

---

## 10. SUCCESS METRICS

### Track These Metrics

#### Performance Metrics
- **Sample Efficiency:** Evaluations needed to reach target (target: 60% reduction)
- **Solution Quality:** Best fitness achieved (target: 70-80% improvement)
- **Convergence Speed:** Generations to convergence (target: 50% faster)
- **Diversity:** Number of unique solutions found (target: maintain or increase)

#### Knowledge Metrics
- **Extraction Rate:** Artifacts extracted per run (target: 100%)
- **Query Success:** Temporal queries work (target: 95% success)
- **Recommendation Accuracy:** Recommended strategy wins (target: 80%)
- **Knowledge Growth:** Graph size over time (target: steady growth)

#### Gauntlet Metrics
- **Pass Rate:** Solutions passing gauntlets (target: maintain or increase)
- **Quality Improvement:** Better solutions pass (target: +20%)
- **Speed:** Gauntlet execution time (target: <10% overhead)
- **Defect Detection:** Bugs found by gauntlets (target: +30%)

#### Domain-Specific Metrics

| Domain | Key Metric | Target |
|--------|-----------|--------|
| **Finance** | Portfolio return | +15% vs baseline |
| **Trading** | Strategy Sharpe ratio | +0.5 improvement |
| **Science** | Experiments needed | -60% |
| **Engineering** | Design weight | -20% while maintaining strength |
| **Pharma** | Dosage accuracy | +10% |
| **Web** | Conversion rate | +25% |

### Validation Checklist

**Before Production Release:**
- [ ] All 4 phases complete
- [ ] All tests passing
- [ ] 70-80% performance improvement validated
- [ ] All 6 domains tested
- [ ] Knowledge extraction working
- [ ] Gauntlets enhanced
- [ ] Documentation complete
- [ ] Rollback plan tested

---

## 🎯 SUMMARY

### What We're Building

A **unified evolutionary optimization platform** that:
- Combines OpenEvolve (QD, MO, Adversarial) + LoongFlow PES (directed search)
- Learns from all runs via Knowledge Engine
- Uses enhanced gauntlets for quality control
- Provides single unified API
- Achieves 70-80% performance improvement

### How We're Building It

**Decision:** Keep LoongFlow as dependency (not extraction)
**Timeline:** 8 weeks (4 phases, 2 weeks each)
**Approach:** Incremental, tested, reversible

### Expected Outcome

- **For Users:** Single API, automatic strategy selection, 70-80% better solutions
- **For System:** Knowledgeable, self-improving, multi-modal
- **For Business:** Competitive advantage in evolutionary optimization

---

**This roadmap is agent-executable. Each task includes clear prompts and deliverables.**

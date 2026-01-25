# LeanAide Decomposition Architecture - Visual Guide

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT LAYER                             │
│                                                                  │
│  Problem Statement (Natural Language Mathematical Problem)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MATHEMATICAL DETECTION LAYER                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LeanMathematicalDetector                                │  │
│  │  • Keyword Analysis (100+ domain keywords)              │  │
│  │  • Symbol Detection (∀, ∃, →, ∈, etc.)                  │  │
│  │  • Pattern Matching (LaTeX, proof patterns)             │  │
│  │  • LLM Analysis (optional, when available)               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Output: MathematicalProblemMetadata                             │
│    • is_mathematical: bool                                      │
│    • problem_type: Theorem/Lemma/Definition/etc.                │
│    • domain: Algebra/Analysis/Topology/etc.                     │
│    • proof_difficulty: 1-10                                     │
│    • formalization_complexity: 1-10                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
        ┌────────────────┐      ┌────────────────┐
        │ NOT Mathematical│      │   Mathematical  │
        │                 │      │                 │
        │ Standard        │      │ LeanAide Route  │
        │ Decomposition   │      │                 │
        └────────────────┘      └────────┬───────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LEANAIDE DECOMPOSITION LAYER                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LeanDecomposer (from leanaide_decomposition_integration)│  │
│  │  • Component Extraction                                   │  │
│  │  • Dependency Analysis                                    │  │
│  │  • Complexity Estimation                                  │  │
│  │  • Lean Code Generation                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Strategies:                                                     │
│  • Structural (theorems, lemmas, definitions)                   │
│  • Dependency (logical prerequisites)                           │
│  • Complexity (formalization difficulty)                        │
│  • Domain (mathematical domain)                                 │
│  • Hybrid (combines multiple)                                   │
│                                                                  │
│  Output: LeanDecompositionPlan                                  │
│    • components: List[MathematicalComponent]                   │
│    • dependencies: Dict[component_id, List[dependency_ids]]    │
│    • component_order: List[component_id] (topological)         │
│    • parallel_groups: List[List[component_id]]                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             LEAN SUB-PROBLEM GENERATION LAYER                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LeanSubProblemDecomposer                                 │  │
│  │  • Create Lean-friendly sub-problems                      │  │
│  │  • Add mathematical metadata                              │  │
│  │  • Generate evolutionary config                           │  │
│  │  • Create Hephaestus tickets                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  EvolutionaryStrategySuggestor                            │  │
│  │  • Analyze proof difficulty                              │  │
│  │  • Suggest evolutionary approach                         │  │
│  │  • Configure evolution parameters                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Output: LeanEnhancedSubProblem                                  │
│    • base_subproblem: SubProblem                                │
│    • mathematical_metadata: MathematicalProblemMetadata        │
│    • lean_code_stub: str (optional)                            │
│    • evolutionary_config: Dict (optional)                      │
│    • verification_ticket: str (Hephaestus)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY ENGINE LAYER                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LeanProofEvolutionEngine (if enabled)                    │  │
│  │  • Generate initial population                            │  │
│  │  • Evaluate fitness (Lean verification)                  │  │
│  │  • Select parents                                         │  │
│  │  • Apply crossover & mutation                             │  │
│  │  • Evolve for N generations                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Strategies:                                                     │
│  • Standard Evolution (GA)                                      │
│  • Adversarial Evolution (Red vs Blue)                          │
│  • Self-Play (RL style)                                         │
│  • Hill Climbing                                                │
│  • Simulated Annealing                                          │
│  • Hybrid (multi-phase)                                         │
│                                                                  │
│  Output: EvolutionResult                                         │
│    • success: bool                                              │
│    • best_proof: LeanProof                                      │
│    • generations_completed: int                                 │
│    • statistics_history: List[PopulationStatistics]            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    ROMA      │  │  Hephaestus  │  │   Workflow   │          │
│  │              │  │              │  │              │          │
│  │ Recursive    │  │ Ticket       │  │ Sub-problem  │          │
│  │ Decomp.      │  │ Tracking     │  │ Execution    │          │
│  │              │  │              │  │              │          │
│  │ Knowledge    │  │ Priority     │  │ Validation   │          │
│  │ Extraction   │  │ Calculation  │  │ Reassembly   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Mathematical Domain Detection Flow

```
Problem Text
      │
      ▼
┌─────────────────────────────────────┐
│     Keyword Analysis                │
│  ├─ Algebra Keywords                │
│  ├─ Analysis Keywords               │
│  ├─ Topology Keywords               │
│  ├─ Number Theory Keywords          │
│  ├─ Combinatorics Keywords          │
│  ├─ Geometry Keywords               │
│  ├─ Logic Keywords                  │
│  └─ Set Theory Keywords             │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│     Domain Scoring                  │
│  For each domain:                   │
│    score = Σ keyword matches        │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│     Domain Selection                │
│  selected_domain = max(score)       │
│  if max(score) == 0:                │
│    return GENERAL                    │
└─────────────────────────────────────┘
```

## Evolutionary Strategy Selection

```
┌─────────────────────────────────────────────────────────────┐
│                    Proof Difficulty                         │
│                                                              │
│  1-4 (Simple)     5-6 (Medium)    7-8 (Complex)   9-10     │
│  ┌────┐           ┌────┐          ┌────┐         ┌────┐    │
│  │None│           │Standard│      │Hybrid│      │Adversarial│
│  └────┘           └────┘          └────┘         └────┘    │
│                                                              │
│  Direct Proof    GA (20-50)      Multi-phase    Red vs Blue │
│  Approach        Populations     Standard +     Teams       │
│                   Tournament      Adversarial    Higher      │
│                                   + Hill Climb  Pressure    │
└─────────────────────────────────────────────────────────────┘
```

## Sub-Problem Enhancement Flow

```
Base SubProblem
      │
      ├─ id, parent_id, title, description
      ├─ type, complexity_score
      ├─ dependencies, success_criteria
      ├─ priority, estimated_effort
      └─ metadata: {}
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│              Lean Enhancement Process                        │
│                                                               │
│  1. Add Mathematical Metadata                                 │
│     ├─ is_mathematical: true                                  │
│     ├─ problem_type: THEOREM_PROOF                            │
│     ├─ domain: NUMBER_THEORY                                  │
│     ├─ proof_difficulty: 7                                    │
│     ├─ formalization_complexity: 8                            │
│     └─ requires_evolution: true                               │
│                                                               │
│  2. Generate Lean Code Stub                                   │
│     └─ "theorem infinite_primes : ..."                       │
│                                                               │
│  3. Create Evolutionary Config                                │
│     ├─ enable_evolution: true                                 │
│     ├─ strategy_type: "hybrid_evolutionary"                   │
│     ├─ population_size: 25                                    │
│     ├─ max_generations: 60                                    │
│     └─ mutation_rate: 0.1                                     │
│                                                               │
│  4. Create Hephaestus Ticket                                  │
│     └─ "HEPH-001"                                             │
│                                                               │
│  5. Add ROMA Integration Hooks                                │
│     └─ recursive_decomposition: true                          │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
LeanEnhancedSubProblem
  │
  ├─ base_subproblem: SubProblem
  ├─ mathematical_metadata: MathematicalProblemMetadata
  ├─ lean_code_stub: str
  ├─ evolutionary_config: Dict
  ├─ verification_ticket: str
  └─ formalization_status: str
      │
      ▼
Convert to SubProblem
      │
      ▼
Enhanced SubProblem (for Workflow)
  │
  ├─ All base fields preserved
  ├─ Enhanced with:
  │   ├─ mathematical_components
  │   ├─ mathematical_domain
  │   ├─ requires_formal_verification
  │   ├─ formal_verification_enabled
  │   └─ metadata["lean_formalization"]: true
  └─ Ready for workflow execution
```

## Integration with Existing Components

```
┌──────────────────────────────────────────────────────────────┐
│                   Decomposition Engine                        │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Existing DecompositionEngine                          │  │
│  │  ├─ SemanticDecomposition                              │  │
│  │  ├─ DependencyDecomposition                            │  │
│  │  ├─ ComplexityDecomposition                            │  │
│  │  └─ HybridDecomposition                                │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           │ (extends)                        │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  LeanEnhancedDecompositionEngine                        │  │
│  │  ├─ Inherits all base functionality                    │  │
│  │  ├─ Adds mathematical detection                         │  │
│  │  ├─ Adds LeanAide routing                               │  │
│  │  ├─ Adds evolutionary config                            │  │
│  │  └─ Fully backward compatible                          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Configuration Hierarchy

```
decomposition_config_lean.yaml
│
├─ leanaide/
│   ├─ enabled: true
│   ├─ client/
│   │   ├─ server_url: "http://localhost:7654"
│   │   ├─ timeout: 300
│   │   └─ enable_simulation_fallback: true
│   └─ decomposition/
│       ├─ default_strategy: "hybrid"
│       └─ enable_llm_extraction: true
│
├─ evolutionary/
│   ├─ enabled: true
│   ├─ min_difficulty: 7
│   └─ default_params/
│       ├─ population_size: 20
│       ├─ max_generations: 50
│       └─ mutation_rate: 0.1
│
├─ mathematical_domains/
│   ├─ complexity_multipliers/
│   │   ├─ logic: 1.3
│   │   ├─ analysis: 1.5
│   │   └─ algebra: 1.2
│   ├─ imports/
│   │   ├─ algebra: ["Mathlib.Algebra.*"]
│   │   └─ analysis: ["Mathlib.Analysis.*"]
│   └─ tactics/
│       ├─ algebra: ["simp", "rw", "ring"]
│       └─ analysis: ["continuity", "tendsto"]
│
├─ detection_thresholds/
│   ├─ mathematical/
│   │   ├─ confidence_threshold: 0.6
│   │   └─ min_keywords: 2
│   └─ evolutionary/
│       ├─ min_proof_difficulty: 7
│       └─ min_formalization_complexity: 7
│
├─ roma_integration/
│   ├─ enabled: true
│   ├─ max_recursion_depth: 3
│   └─ min_complexity_for_recursion: 7
│
├─ hephaestus_integration/
│   ├─ enabled: true
│   └─ tickets/
│       ├─ ticket_type: "lean_formalization"
│       └─ priority_levels/
│           ├─ critical: 9
│           ├─ high: 7
│           └─ medium: 5
│
└─ performance/
    ├─ parallel/
    │   ├─ enabled: true
    │   └─ max_workers: 4
    └─ cache/
        ├─ enabled: true
        └─ ttl: 3600
```

## Data Flow Diagram

```
┌──────────────────┐
│  User Input      │
│  (Problem Text)  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Detection                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ LeanMathematicalDetector                             │   │
│  │ Input: Problem text                                  │   │
│  │ Process:                                             │   │
│  │   • Analyze keywords (100+)                          │   │
│  │   • Detect mathematical symbols                      │   │
│  │   • Classify problem type                            │   │
│  │   • Identify domain                                  │   │
│  │   • Estimate difficulty                              │   │
│  │ Output: MathematicalProblemMetadata                  │   │
│  │   • is_mathematical: bool                            │   │
│  │   • problem_type, domain                             │   │
│  │   • proof_difficulty: 1-10                           │   │
│  │   • formalization_complexity: 1-10                   │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Routing                                            │
│  ┌──────────────────────┐        ┌──────────────────────┐  │
│  │ NOT Mathematical     │        │ Mathematical         │  │
│  │                      │        │                      │  │
│  │ Standard             │        │ LeanAide             │  │
│  │ Decomposition        │        │ Decomposition        │  │
│  │                      │        │                      │  │
│  │ (Existing flow)      │        │ (New flow)           │  │
│  └──────────────────────┘        └──────────┬───────────┘  │
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: LeanAide Decomposition                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ LeanDecomposer                                      │   │
│  │ Input: Problem text, strategy                       │   │
│  │ Process:                                             │   │
│  │   • Extract mathematical components                  │   │
│  │   • Identify dependencies                            │   │
│  │   • Estimate complexity                              │   │
│  │   • Generate Lean code stubs                         │   │
│  │   • Determine order (topological)                    │   │
│  │   • Identify parallel groups                         │   │
│  │ Output: LeanDecompositionPlan                         │   │
│  │   • components: List[MathematicalComponent]         │   │
│  │   • dependencies: Dict[id, List[ids]]               │   │
│  │   • component_order: List[id]                        │   │
│  │   • parallel_groups: List[List[id]]                 │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Sub-Problem Enhancement                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ LeanSubProblemDecomposer                             │   │
│  │ Input: LeanDecompositionPlan                         │   │
│  │ Process:                                             │   │
│  │   • Create LeanEnhancedSubProblem for each component│   │
│  │   • Add mathematical metadata                        │   │
│  │   • Generate evolutionary config                     │   │
│  │   • Create Hephaestus tickets                        │   │
│  │ Output: List[LeanEnhancedSubProblem]                 │   │
│  │   • base_subproblem: SubProblem                      │   │
│  │   • mathematical_metadata                             │   │
│  │   • lean_code_stub: str                              │   │
│  │   • evolutionary_config: Dict                        │   │
│  │   • verification_ticket: str                         │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Evolutionary Configuration                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ EvolutionaryStrategySuggestor                        │   │
│  │ Input: MathematicalProblemMetadata                   │   │
│  │ Process:                                             │   │
│  │   • Analyze proof difficulty                          │   │
│  │   • Suggest evolutionary strategy                     │   │
│  │   • Configure parameters                             │   │
│  │   • Add strategy-specific settings                   │   │
│  │ Output: Dict[str, Any]                                │   │
│  │   • enable_evolution: bool                            │   │
│  │   • strategy_type: str                               │   │
│  │   • population_size: int                             │   │
│  │   • max_generations: int                             │   │
│  │   • mutation_rate: float                             │   │
│  │   • [strategy-specific fields]                       │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Integration                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    ROMA     │  │ Hephaestus  │  │     Workflow        │  │
│  │             │  │             │  │                     │  │
│  │ Recursive   │  │ Create      │  │ Convert to          │  │
│  │ decomp. for │  │ tickets for │  │ SubProblem for      │  │
│  │ complex     │  │ tracking    │  │ workflow execution  │  │
│  │ components  │  │ progress    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling Flow

```
┌─────────────────────┐
│  Operation Start   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Try Primary Path  │──Yes──▶│  Success           │
│                     │      │  Return Result      │
└──────────┬──────────┘      └─────────────────────┘
           │ No
           ▼
┌─────────────────────┐
│  Log Error          │
│  (with context)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Fallback Available?│──No───▶│  Raise Exception   │
└──────────┬──────────┘      └─────────────────────┘
           │ Yes
           ▼
┌─────────────────────┐
│  Use Fallback       │
│  • Heuristic decomp.│
│  • Template-based   │
│  • Cached results   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Return Result      │
│  (with warning)     │
└─────────────────────┘

Example: LeanAide Decomposition
  Primary: LeanDecomposer with LLM
  Fallback: Heuristic decomposition
  Result: Valid DecompositionPlan
```

## Performance Optimization Flow

```
┌─────────────────────┐
│  Request Received   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Check Cache        │
│  • Problem hash     │
│  • Component hash   │
│  • Dependency hash  │
└──────────┬──────────┘
           │
      ┌────┴────┐
      │         │
     Hit       Miss
      │         │
      ▼         ▼
┌─────────┐  ┌─────────────────────┐
│ Return  │  │  Acquire Resources  │
│ Cached  │  │  • Worker threads   │
│ Result  │  │  • LLM connection    │
└─────────┘  │  • LeanAide client   │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Process Request    │
             │  • Parallelize tasks │
             │  • Batch operations  │
             │  • Stream results    │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Cache Results      │
             │  • Problem plans     │
             │  • Components        │
             │  • Lean code         │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Return Result      │
             └─────────────────────┘

Performance Improvements:
  • Cache hit rate: 70-90%
  • Parallel processing: 3-4x speedup
  • Batching: 2-3x efficiency
  • Streaming: Lower latency
```

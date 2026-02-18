"""
LeanAide Decomposition Engine - Quick Reference Guide

==============================================
DETECTION AND ROUTING
==============================================

# Detect if a problem is mathematical
from decomposition_engine_lean_enhanced import detect_and_route_mathematical_problem

plan, math_metadata = await detect_and_route_mathematical_problem(problem)

if math_metadata.is_mathematical:
    print(f"Domain: {math_metadata.domain.value}")
    print(f"Type: {math_metadata.problem_type.value}")
    print(f"Difficulty: {math_metadata.proof_difficulty}/10")
    print(f"Strategy: {math_metadata.recommended_evolutionary_strategy.value}")

==============================================
LEAN-ENHANCED DECOMPOSITION
==============================================

# Create engine with LeanAide integration
from decomposition_engine_lean_enhanced import LeanEnhancedDecompositionEngine

engine = LeanEnhancedDecompositionEngine(
    enable_lean_detection=True,
    enable_evolution=True,
    leanaide_decomposer=leanaide_decomposer  # Optional
)

# Decompose with automatic routing
plan = await engine.decompose_with_leanaide(problem)

# Check if Lean decomposition was used
if plan.metadata.get("lean_decomposition"):
    print("Used LeanAide-specific decomposition")
    domain = plan.metadata.get("mathematical_domain")
    print(f"Domain: {domain}")

==============================================
MATHEMATICAL PROBLEM METADATA
==============================================

from decomposition_engine_lean_enhanced import LeanMathematicalDetector

detector = LeanMathematicalDetector(enable_llm=True)
metadata = detector.detect_mathematical_problem(
    "Prove that sqrt(2) is irrational",
    "Irrationality Proof"
)

# Access metadata
metadata.is_mathematical          # bool
metadata.problem_type             # MathematicalProblemType
metadata.domain                   # MathematicalDomain
metadata.proof_difficulty         # int 1-10
metadata.formalization_complexity # int 1-10
metadata.recommended_evolutionary_strategy  # EvolutionaryStrategyType
metadata.requires_evolution       # bool

==============================================
EVOLUTIONARY CONFIGURATION
==============================================

from decomposition_engine_lean_enhanced import generate_evolutionary_config

config = await generate_evolutionary_config(math_metadata)

# Config structure
{
    "enable_evolution": True,
    "strategy_type": "standard_evolution",
    "population_size": 20,
    "max_generations": 50,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "selection_method": "tournament",
    "crossover_method": "uniform",
    "elitism_ratio": 0.1
    # Strategy-specific fields...
}

# Apply to LeanAide evolution engine
from leanaide_evolution import LeanProofEvolutionEngine

evolution_engine = LeanProofEvolutionEngine(
    theorem=problem_statement,
    **config
)

result = await evolution_engine.evolve()

==============================================
LEAN-ENHANCED SUB-PROBLEMS
==============================================

from decomposition_engine_lean_enhanced import LeanEnhancedSubProblem

# Create enhanced sub-problem
enhanced_sp = LeanEnhancedSubProblem(
    base_subproblem=workflow_subproblem,
    mathematical_metadata=math_metadata,
    lean_code_stub="theorem infinite_primes : ...",
    evolutionary_config=evolution_config,
    verification_ticket="HEPH-001",
    formalization_status="pending"
)

# Convert to standard SubProblem for workflow
workflow_sp = enhanced_sp.to_subproblem()

# Access Lean-specific fields
enhanced_sp.mathematical_metadata
enhanced_sp.lean_code_stub
enhanced_sp.evolutionary_config
enhanced_sp.verification_ticket

==============================================
MATHEMATICAL DOMAINS
==============================================

from leanaide_decomposition_integration import MathematicalDomain

# Available domains
MathematicalDomain.ALGEBRA
MathematicalDomain.ANALYSIS
MathematicalDomain.TOPOLOGY
MathematicalDomain.NUMBER_THEORY
MathematicalDomain.COMBINATORICS
MathematicalDomain.GEOMETRY
MathematicalDomain.LOGIC
MathematicalDomain.SET_THEORY
MathematicalDomain.GENERAL

# Domain-specific imports
DOMAIN_IMPORTS = {
    MathematicalDomain.ALGEBRA: ["Mathlib.Algebra.*", "Mathlib.Data.*"],
    MathematicalDomain.ANALYSIS: ["Mathlib.Analysis.*", "Mathlib.Topology.*"],
    MathematicalDomain.NUMBER_THEORY: ["Mathlib.Data.Nat.*", "Mathlib.NumberTheory.*"],
    # ... etc
}

==============================================
EVOLUTIONARY STRATEGIES
==============================================

from decomposition_engine_lean_enhanced import EvolutionaryStrategyType

# Available strategies
EvolutionaryStrategyType.STANDARD_EVOLUTION
EvolutionaryStrategyType.ADVERSARIAL_EVOLUTION
EvolutionaryStrategyType.SELF_PLAY
EvolutionaryStrategyType.HILL_CLIMBING
EvolutionaryStrategyType.SIMULATED_ANNEALING
EvolutionaryStrategyType.HYBRID_EVOLUTIONARY

# Strategy selection guide
DIFFICULTY_1_4 → None (direct proof)
DIFFICULTY_5_6 → Standard Evolution
DIFFICULTY_7_8 → Hybrid Evolutionary
DIFFICULTY_9_10 → Adversarial Evolution

==============================================
PROBLEM TYPES
==============================================

from decomposition_engine_lean_enhanced import MathematicalProblemType

# Available types
MathematicalProblemType.THEOREM_PROOF
MathematicalProblemType.LEMMA_PROOF
MathematicalProblemType.DEFINITION_FORMALIZATION
MathematicalProblemType.CONJECTURE_INVESTIGATION
MathematicalProblemType.EXERCISE_SOLUTION
MathematicalProblemType.CONSTRUCTION_PROBLEM
MathematicalProblemType.COMPUTATION_PROBLEM
MathematicalProblemType.GENERAL_MATHEMATICS

# Base difficulty by type
DEFINITION_FORMALIZATION: 3
EXERCISE_SOLUTION: 4
LEMMA_PROOF: 5
THEOREM_PROOF: 6
CONSTRUCTION_PROBLEM: 7
CONJECTURE_INVESTIGATION: 9

==============================================
LEAN COMPONENT TYPES
==============================================

from leanaide_decomposition_integration import ComponentType

# Available component types
ComponentType.THEOREM
ComponentType.LEMMA
ComponentType.DEFINITION
ComponentType.PROPOSITION
ComponentType.COROLLARY
ComponentType.EXAMPLE
ComponentType.AXIOM
ComponentType.CONJECTURE
ComponentType.EXERCISE
ComponentType.REMARK

# Formalization complexity by type
DEFINITION: 3 (easiest)
COROLLARY: 3
LEMMA: 4
PROPOSITION: 5
THEOREM: 6
CONJECTURE: 8 (hardest)

==============================================
CONFIGURATION
==============================================

# Load configuration
import yaml

with open("decomposition_config_lean.yaml") as f:
    config = yaml.safe_load(f)

# Key configuration sections
config["leanaide"]["enabled"]              # Enable LeanAide integration
config["evolutionary"]["enabled"]          # Enable evolutionary approach
config["detection_thresholds"]             # Detection thresholds
config["mathematical_domains"]             # Domain settings
config["performance"]["parallel"]["enabled"] # Parallel processing

# Update configuration
config["leanaide"]["decomposition"]["default_strategy"] = "dependency"
config["evolutionary"]["min_difficulty"] = 6

==============================================
INTEGRATION WITH WORKFLOW
==============================================

# Convert decomposition plan to workflow sub-problems
from decomposition_engine_lean_enhanced import LeanEnhancedDecompositionEngine

engine = LeanEnhancedDecompositionEngine()
plan = await engine.decompose_with_leanaide(problem)

# Access sub-problems
for sub_problem in plan.sub_problems:
    # Each sub_problem is a standard SubProblem
    # with enhanced metadata

    # Check if it's a Lean formalization task
    if sub_problem.metadata.get("lean_formalization"):
        print(f"Lean task: {sub_problem.title}")

        # Access domain
        domain = sub_problem.mathematical_domain
        print(f"Domain: {domain.value if domain else 'unknown'}")

        # Access complexity
        complexity = sub_problem.complexity_score.overall_complexity
        print(f"Complexity: {complexity}/10")

        # Get evolutionary config
        evo_config = sub_problem.metadata.get("evolutionary_config")
        if evo_config:
            print(f"Strategy: {evo_config['strategy_type']}")

==============================================
CREWAI TICKET CREATION
==============================================

# Tickets are auto-created if enabled in config
# Configure:
crewai_integration:
  enabled: true
  tickets:
    ticket_type: "lean_formalization"
    priority_levels:
      critical: 9
      high: 7
      medium: 5
      low: 3

# Access ticket ID from sub-problem
ticket_id = sub_problem.metadata.get("verification_ticket")
if ticket_id:
    print(f"CrewAI ticket: {ticket_id}")

==============================================
ROMA INTEGRATION
==============================================

# ROMA is used for recursive decomposition
# Configure:
roma_integration:
  enabled: true
  max_recursion_depth: 3
  min_complexity_for_recursion: 7

# When a sub-problem has complexity >= 7,
# it will be recursively decomposed using ROMA

if sub_problem.complexity_score.overall_complexity >= 7:
    # Trigger ROMA recursive decomposition
    roma_subproblems = await roma.decompose(sub_problem)

==============================================
COMMON WORKFLOWS
==============================================

# Workflow 1: Simple mathematical problem
problem = ProblemDefinition(...)
plan, metadata = await detect_and_route_mathematical_problem(problem)

if metadata.is_mathematical:
    print(f"Mathematical problem detected: {metadata.domain.value}")
    print(f"Proof difficulty: {metadata.proof_difficulty}/10")

# Workflow 2: Full LeanAide decomposition
engine = LeanEnhancedDecompositionEngine(
    enable_lean_detection=True,
    enable_evolution=True
)

plan = await engine.decompose_with_leanaide(problem)

for sp in plan.sub_problems:
    if sp.metadata.get("lean_formalization"):
        # Generate evolutionary config
        config = await generate_evolutionary_config(
            sp.metadata.get("mathematical_metadata")
        )

        # Run evolution if needed
        if config.get("enable_evolution"):
            evolution_engine = LeanProofEvolutionEngine(
                theorem=sp.description,
                **config
            )
            result = await evolution_engine.evolve()

# Workflow 3: Integration with existing decomposition
# Just replace DecompositionEngine with LeanEnhancedDecompositionEngine
# No other changes needed - backward compatible

==============================================
ERROR HANDLING
==============================================

# All LeanAide operations have graceful fallbacks

# If LeanAide is not available:
# - Falls back to heuristic decomposition
# - Mathematical detection still works
# - Evolutionary config still generated

# If decomposition fails:
# - Falls back to single-component decomposition
# - Logs error but continues
# - Returns valid DecompositionPlan

# If evolution fails:
# - Returns result with success=False
# - Includes error details
# - Can retry with different parameters

==============================================
LOGGING
==============================================

import logging

# Enable LeanAide decomposition logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decomposition_engine_lean_enhanced")
logger.setLevel(logging.DEBUG)

# Key log messages:
# "Mathematical detection: is_mathematical=True"
# "Using LeanAide-specific decomposition for mathematical problem"
# "LeanAide decomposition complete: N sub-problems"
# "Evolutionary configuration generated: strategy=X"

==============================================
TESTING
==============================================

# Test detection
async def test_detection():
    detector = LeanMathematicalDetector()
    metadata = detector.detect_mathematical_problem(
        "Prove that there are infinitely many primes"
    )
    assert metadata.is_mathematical
    assert metadata.domain == MathematicalDomain.NUMBER_THEORY

# Test decomposition
async def test_decomposition():
    engine = LeanEnhancedDecompositionEngine()
    plan = await engine.decompose_with_leanaide(problem)
    assert len(plan.sub_problems) > 0
    assert plan.metadata.get("lean_decomposition") == True

# Test evolutionary config
async def test_evolutionary_config():
    config = await generate_evolutionary_config(metadata)
    assert config["enable_evolution"] == True
    assert "population_size" in config

==============================================
PERFORMANCE TIPS
==============================================

1. Enable caching (default: enabled)
   - Reduces redundant computations
   - Stores decomposition results
   - Caches Lean code generation

2. Use parallel processing (default: enabled)
   - Parallel component extraction
   - Parallel complexity estimation
   - Parallel Lean code generation

3. Adjust timeouts for large problems
   - decomposition: 60s (default)
   - component_extraction: 30s
   - complexity_estimation: 10s

4. Use appropriate detection thresholds
   - Lower thresholds = more false positives
   - Higher thresholds = miss some mathematical problems
   - Default: confidence_threshold = 0.6

==============================================
TROUBLESHOOTING
==============================================

# Problem: Mathematical problems not detected
# Solution: Lower confidence threshold
detector = LeanMathematicalDetector()
metadata = detector.detect_mathematical_problem(text)
# Adjust: config["detection_thresholds"]["mathematical"]["confidence_threshold"] = 0.4

# Problem: LeanAide decomposition fails
# Solution: Check LeanAide server and fallback is automatic
# Fallback to heuristic decomposition

# Problem: Evolutionary config not generated
# Solution: Check problem difficulty exceeds threshold
# config["evolutionary"]["min_difficulty"] = 6

# Problem: Lean code generation fails
# Solution: Lean code is optional
# config["subproblem_generation"]["include_code_stubs"] = False
"""

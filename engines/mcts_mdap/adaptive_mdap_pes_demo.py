"""
Demo and Usage Examples for Adaptive MDAP + PES Integration
===========================================================

This file demonstrates how to use the adaptive_mdap_pes_integration module
to achieve 40-60% cost savings over standalone systems.

Run with:
    python adaptive_mdap_pes_demo.py
"""
from __future__ import annotations


import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_1_basic_usage():
    """
    Demo 1: Basic usage of the Adaptive PES Coordinator.
    
    Shows the simplest way to use the integrated system.
    """
    print("\n" + "="*70)
    print("DEMO 1: Basic Usage")
    print("="*70)
    
    print("""
from adaptive_mdap_pes_integration import AdaptivePESCoordinator

# Create coordinator with $10 budget
coordinator = AdaptivePESCoordinator(max_budget_usd=10.0)

# Optimize code
result = await coordinator.optimize(
    problem_description="Optimize Python sorting algorithm",
    code="def sort(arr): return sorted(arr)",
    tests=[{"input": "[3,1,2]", "expected": "[1,2,3]"}],
    language="python"
)

# Access results
print(f"Cost: ${result.total_cost_usd:.2f}")
print(f"Efficiency gain: {result.efficiency_gain:.0%}")
print(f"Complexity: {result.complexity_analysis.overall_score:.3f}")
print(f"Tier: {result.allocation_decision.tier.value}")
    """)


def demo_2_cost_estimation():
    """
    Demo 2: Pre-flight cost estimation.
    
    Shows how to estimate costs before running optimization.
    """
    print("\n" + "="*70)
    print("DEMO 2: Cost Estimation")
    print("="*70)
    
    print("""
from adaptive_mdap_pes_integration import create_cost_aware_coordinator

# Create cost-aware coordinator
coordinator = create_cost_aware_coordinator(max_budget_usd=5.0)

# Get cost estimate
estimate = coordinator.get_cost_estimate(
    problem_description="Build REST API endpoint for user authentication",
    code="# Flask API code here",
    language="python"
)

print(f"Estimated complexity: {estimate['estimated_complexity']:.3f}")
print(f"Recommended tier: {estimate['recommended_tier']}")

# Show estimates for all tiers
print("\\nCost estimates by tier:")
for tier, est in estimate['tier_estimates'].items():
    print(f"  {tier}: ${est['estimated_cost_usd']:.2f} "
          f"({est['estimated_evaluations']} evals)")

# Output might look like:
# Estimated complexity: 0.650
# Recommended tier: maker_full
# 
# Cost estimates by tier:
#   direct: $0.50 (200 evals)
#   mdap_light: $1.50 (1000 evals)
#   mdap_medium: $2.50 (1500 evals)
#   maker_full: $4.00 (2000 evals)
#   maker_ultra: $6.00 (2500 evals) [exceeds budget]
    """)


def demo_3_allocation_recommendation():
    """
    Demo 3: Getting allocation recommendations without executing.
    
    Useful for UI display and pre-flight checks.
    """
    print("\n" + "="*70)
    print("DEMO 3: Allocation Recommendation")
    print("="*70)
    
    print("""
from adaptive_mdap_pes_integration import AdaptivePESCoordinator

coordinator = AdaptivePESCoordinator()

# Get recommendation without executing
allocation = coordinator.get_allocation_recommendation(
    problem_description="Implement distributed consensus algorithm",
    code="# Complex distributed systems code",
    language="python",
    budget_remaining_pct=75.0
)

print(f"Complexity score: {allocation.complexity_score:.3f}")
print(f"Recommended tier: {allocation.tier.value}")
print(f"Agents: {allocation.n_agents}")
print(f"K ahead: {allocation.k_ahead}")
print(f"Max retries: {allocation.max_retries}")
print(f"Timeout: {allocation.timeout_ms}ms")
print(f"Estimated cost: ${allocation.estimated_cost_usd:.2f}")
print(f"Estimated evaluations: {allocation.estimated_evaluations}")
print(f"PES strategy: {allocation.pes_strategy.value if allocation.pes_strategy else 'None'}")
print("\\nReasoning:")
for reason in allocation.reasoning:
    print(f"  * {reason}")
    """)


def demo_4_different_configurations():
    """
    Demo 4: Using different pre-configured setups.
    
    Shows cost-focused vs performance-focused configurations.
    """
    print("\n" + "="*70)
    print("DEMO 4: Different Configurations")
    print("="*70)
    
    print("""
from adaptive_mdap_pes_integration import (
    create_cost_aware_coordinator,
    create_performance_coordinator,
    create_fully_featured_coordinator,
    AdaptivePESConfig
)

# Option 1: Cost-focused (minimum spend)
cost_coordinator = create_cost_aware_coordinator(max_budget_usd=3.0)

# Option 2: Performance-focused (maximize quality)
perf_coordinator = create_performance_coordinator(max_budget_usd=25.0)

# Option 3: Fully featured (all capabilities enabled)
full_coordinator = create_fully_featured_coordinator(max_budget_usd=10.0)

# Option 4: Custom configuration
custom_config = AdaptivePESConfig(
    max_budget_usd=15.0,
    complexity_thresholds=[0.15, 0.35, 0.55, 0.75],  # More aggressive
    enable_adaptive_allocation=True,
    enable_context_aware=True,
    enable_early_stopping=True,
    unified_budget_tracking=True,
    cross_system_learning=True
)
custom_coordinator = AdaptivePESCoordinator(config=custom_config)

# Use any coordinator the same way
result = await cost_coordinator.optimize(...)
    """)


def demo_5_workflow_integration():
    """
    Demo 5: Integration with workflow_engine.py.
    
    Shows how the integration connects to existing systems.
    """
    print("\n" + "="*70)
    print("DEMO 5: Workflow Engine Integration")
    print("="*70)
    
    print("""
# In workflow_engine.py, the integration provides:

from adaptive_mdap_pes_integration import AdaptivePESCoordinator

class WorkflowEngine:
    def __init__(self):
        self.adaptive_pes = AdaptivePESCoordinator()
    
    async def execute_gauntlet_with_adaptive_pes(
        self,
        workflow_state: WorkflowState,
        budget_usd: float = 10.0
    ):
        # Use coordinator for optimization
        result = await self.adaptive_pes.optimize(
            problem_description=workflow_state.problem_description,
            code=workflow_state.code,
            tests=workflow_state.tests,
            language=workflow_state.language,
            max_budget_usd=budget_usd
        )
        
        # Access unified results
        return {
            'solution': result.original_result,
            'cost': result.total_cost_usd,
            'efficiency_gain': result.efficiency_gain,
            'complexity': result.complexity_analysis.overall_score,
            'tier': result.allocation_decision.tier.value,
            'recommendations': result.recommendations
        }

# Usage:
engine = WorkflowEngine()
result = await engine.execute_gauntlet_with_adaptive_pes(
    workflow_state=my_workflow,
    budget_usd=15.0
)
    """)


def demo_6_maker_integration():
    """
    Demo 6: Integration with maker_engine.py.
    
    Shows how MAKER can use adaptive complexity classification.
    """
    print("\n" + "="*70)
    print("DEMO 6: Maker Engine Integration")
    print("="*70)
    
    print("""
# In maker_engine.py, the integration enables:

from adaptive_mdap_pes_integration import (
    AdaptivePESCoordinator,
    ComplexityPESBridge
)

class MakerEngine:
    def solve_with_adaptive_complexity(
        self,
        initial_state,
        step_builder,
        apply_action,
        problem_description,
        code=None
    ):
        # Get coordinator
        coordinator = AdaptivePESCoordinator()
        
        # Get allocation based on complexity
        allocation = coordinator.get_allocation_recommendation(
            problem_description=problem_description,
            code=code
        )
        
        # Adjust MAKER config
        adjusted_config = MakerConfig(
            k_min=allocation.k_ahead,
            k_max=allocation.k_ahead + 2,
            max_votes_per_step=allocation.n_agents * 12,
            timeout_seconds=allocation.timeout_ms // 1000
        )
        
        # Execute with optimized config
        engine = MakerEngine(self.team, adjusted_config)
        return engine.solve(initial_state, step_builder, apply_action)

# Usage:
engine = MakerEngine(team, config)
result = engine.solve_with_adaptive_complexity(
    initial_state=state,
    step_builder=builder,
    apply_action=applier,
    problem_description="Build authentication system",
    code=existing_code
)
    """)


def demo_7_result_analysis():
    """
    Demo 7: Analyzing optimization results.
    
    Shows how to extract insights from the integrated results.
    """
    print("\n" + "="*70)
    print("DEMO 7: Result Analysis")
    print("="*70)
    
    print("""
from adaptive_mdap_pes_integration import AdaptivePESCoordinator

coordinator = AdaptivePESCoordinator()
result = await coordinator.optimize(...)

# 1. Basic metrics
print(f"Total cost: ${result.total_cost_usd:.2f}")
print(f"Efficiency gain: {result.efficiency_gain:.0%}")
print(f"Evaluations saved: {result.evaluations_saved}")
print(f"Execution time: {result.execution_time_ms}ms")

# 2. Complexity analysis
if result.complexity_analysis:
    ca = result.complexity_analysis
    print(f"\\nComplexity Analysis:")
    print(f"  Overall score: {ca.overall_score:.3f}")
    print(f"  Text length: {ca.text_length_score:.3f}")
    print(f"  Domain rarity: {ca.domain_rarity_score:.3f}")
    print(f"  Dependencies: {ca.dependency_score:.3f}")
    print(f"  Keywords: {ca.keyword_score:.3f}")
    print(f"  Confidence: {ca.confidence:.1%}")

# 3. Allocation decision
if result.allocation_decision:
    ad = result.allocation_decision
    print(f"\\nAllocation Decision:")
    print(f"  Tier: {ad.tier.value}")
    print(f"  Agents: {ad.n_agents}")
    print(f"  K ahead: {ad.k_ahead}")
    print(f"  Estimated cost: ${ad.estimated_cost_usd:.2f}")
    print(f"  PES strategy: {ad.pes_strategy.value if ad.pes_strategy else 'None'}")

# 4. Budget status
if result.budget_status:
    bs = result.budget_status
    print(f"\\nBudget Status:")
    print(f"  Used: ${bs.cost_used_usd:.2f} ({bs.cost_pct_used:.1%})")
    print(f"  Remaining: ${bs.cost_remaining_usd:.2f}")
    print(f"  Evaluations: {bs.evaluations_used} used, {bs.evaluations_remaining} remaining")
    print(f"  Status: {bs.status}")

# 5. Execution info
print(f"\\nExecution:")
print(f"  Converged: {result.convergence_achieved}")
print(f"  Stopped early: {result.stopped_early}")
print(f"  Stop reason: {result.stop_reason or 'N/A'}")
print(f"  Phases completed: {[p.value for p in result.phases_completed]}")

# 6. Recommendations
if result.recommendations:
    print(f"\\nRecommendations:")
    for rec in result.recommendations:
        print(f"  * {rec}")

# 7. Convert to dict for serialization
data = result.to_dict()
# Can be JSON-serialized for storage/transmission
    """)


def demo_8_backward_compatibility():
    """
    Demo 8: Backward compatibility with existing APIs.
    
    Shows how existing code can benefit without changes.
    """
    print("\n" + "="*70)
    print("DEMO 8: Backward Compatibility")
    print("="*70)
    
    print("""
# Existing code using openevolve_pes_integration:
from openevolve_pes_integration import enhance_code
result = await enhance_code(code, tests)

# Drop-in replacement with Adaptive PES integration:
from adaptive_mdap_pes_integration import AdaptivePESIntegrationWrapper

wrapper = AdaptivePESIntegrationWrapper(max_budget_usd=10.0)
result = await wrapper.enhance_code(
    code=code,
    problem_description="Optimize code",
    tests=tests,
    language="python"
)

# Same API, but now includes:
# * Complexity-based resource allocation
# * Unified budget tracking
# * 40-60% cost savings
# * Cross-system learning

# The result is a dict with enhanced information:
print(result)
# {
#     'success': True,
#     'best_fitness': 0.95,
#     'total_evaluations': 500,
#     'total_cost_usd': 4.50,
#     'efficiency_gain': 0.45,
#     'evaluations_saved': 400,
#     'converged': True,
#     'stopped_early': False,
#     'complexity_score': 0.65,
#     'allocation_tier': 'maker_full',
#     'recommendations': [...]
# }
    """)


def demo_9_performance_summary():
    """
    Demo 9: Getting performance summaries.
    
    Shows how to track coordinator performance over time.
    """
    print("\n" + "="*70)
    print("DEMO 9: Performance Summary")
    print("="*70)
    
    print("""
from adaptive_mdap_pes_integration import AdaptivePESCoordinator

coordinator = AdaptivePESCoordinator()

# Run multiple optimizations
for problem in problems:
    result = await coordinator.optimize(...)
    # Results are automatically tracked

# Get performance summary
summary = coordinator.get_performance_summary()

print(f"Total executions: {summary['total_executions']}")
print(f"Average efficiency gain: {summary['avg_efficiency_gain']:.1%}")
print(f"Total evaluations saved: {summary['total_evaluations_saved']}")
print(f"Adaptive MDAP available: {summary['adaptive_mdap_available']}")
print(f"PES Enhanced available: {summary['pes_enhanced_available']}")

# Example output:
# Total executions: 50
# Average efficiency gain: 52.3%
# Total evaluations saved: 12500
# Adaptive MDAP available: True
# PES Enhanced available: True
    """)


def demo_10_error_handling():
    """
    Demo 10: Error handling and fallback behavior.
    
    Shows how the system handles failures gracefully.
    """
    print("\n" + "="*70)
    print("DEMO 10: Error Handling")
    print("="*70)
    
    print("""
from adaptive_mdap_pes_integration import AdaptivePESCoordinator, AdaptivePESConfig

# Create coordinator with fallback enabled
config = AdaptivePESConfig(
    fallback_on_error=True,  # Enable fallback
    preserve_existing_behavior=True
)
coordinator = AdaptivePESCoordinator(config=config)

# If components fail, coordinator falls back gracefully
try:
    result = await coordinator.optimize(...)
    
    if result.stopped_early and "fallback" in (result.stop_reason or ""):
        print("Warning: Fallback executed due to error")
        print(f"Reason: {result.stop_reason}")
        # Result still contains safe fallback data
        
except Exception as e:
    # Only raises if fallback_on_error=False
    print(f"Unhandled error: {e}")

# Check which components are available
print(f"Complexity classifier: {coordinator.complexity_classifier is not None}")
print(f"MDAP allocator: {coordinator.mdap_allocator is not None}")
print(f"PES wrapper: {coordinator.pes_wrapper is not None}")
    """)


def print_architecture_diagram():
    """Print the architecture diagram."""
    print("\n" + "="*70)
    print("ARCHITECTURE OVERVIEW")
    print("="*70)
    
    print("""
                    +-------------------------+
                    |   User/Application      |
                    +-------------+-----------+
                                  |
                                  v
                    +-------------------------+
                    | AdaptivePESCoordinator  |
                    |  (Main Entry Point)     |
                    +-------------+-----------+
                                  |
            +---------------------+---------------------+
            |                     |                     |
            v                     v                     v
    +---------------+   +---------------+   +---------------+
    | Adaptive MDAP |   |     Bridge    |   |  PES Enhanced |
    |   System      |<--+   Components  +-->|    System     |
    +---------------+   +---------------+   +---------------+
    | - Complexity  |   | - Complexity  |   | - Cost Optim  |
    |   Classifier  |   |   PESBridge   |   | - Strategy Sel|
    | - Allocator   |   | - UnifiedBudget|  | - Early Stop  |
    |   (5-tier)    |   |   Tracker     |   | - Summarizatn |
    +---------------+   +---------------+   +---------------+
            |                     |                     |
            +---------------------+---------------------+
                                  |
                                  v
                    +-------------------------+
                    |   Existing Systems      |
                    |  - workflow_engine.py   |
                    |  - maker_engine.py      |
                    |  - openevolve_*.py      |
                    +-------------------------+
    """)


def print_data_flow():
    """Print the data flow diagram."""
    print("\n" + "="*70)
    print("DATA FLOW")
    print("="*70)
    
    print("""
Phase 1: COMPLEXITY ANALYSIS
----------------------------
Input -> TaskComplexityClassifier -> ComplexityScore
         (7 features analyzed)      (0-1 score)

Phase 2: ALLOCATION PLANNING
----------------------------
ComplexityScore -> ComplexityPESBridge -> AllocationDecision
                    - complexity_to_tier()    (tier selection)
                    - tier_to_pes_strategy()  (strategy mapping)
                    - complexity_to_params()  (PES parameters)

Phase 3: BUDGET INTEGRATION
---------------------------
AllocationDecision + UnifiedBudgetTracker -> AdjustedDecision
                         - check budget status
                         - adjust if warning/critical
                         - estimate remaining evaluations

Phase 4: EXECUTION
------------------
AdjustedDecision -> PESIntegrationWrapper -> EnhancedEvolutionResult
                     - enhance_with_planning()
                     - cost optimization
                     - early stopping

Phase 5: RESULT AGGREGATION
---------------------------
All Results -> AdaptivePESCoordinator -> AdaptivePESEvolutionResult
                - combine all data
                - generate recommendations
                - calculate efficiency
    """)


def main():
    """Run all demos."""
    print("\n")
    print("+" + "="*68 + "+")
    print("|" + " "*20 + "ADAPTIVE MDAP + PES INTEGRATION" + " "*17 + "|")
    print("|" + " "*15 + "Demo and Usage Examples" + " "*30 + "|")
    print("+" + "="*68 + "+")
    
    print_architecture_diagram()
    print_data_flow()
    
    demo_1_basic_usage()
    demo_2_cost_estimation()
    demo_3_allocation_recommendation()
    demo_4_different_configurations()
    demo_5_workflow_integration()
    demo_6_maker_integration()
    demo_7_result_analysis()
    demo_8_backward_compatibility()
    demo_9_performance_summary()
    demo_10_error_handling()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The Adaptive MDAP + PES Integration provides:

* 40-60% cost reduction vs standalone systems
* Complexity-based resource allocation (5-tier system)
* Unified budget tracking across both systems
* Backward compatibility with existing APIs
* Easy integration with workflow_engine.py and maker_engine.py

Key Components:
* AdaptivePESCoordinator - Main orchestrator
* UnifiedBudgetTracker - Cross-system budget management
* ComplexityPESBridge - Maps complexity to PES strategies
* AdaptivePESIntegrationWrapper - Backward compatibility

For full documentation, see:
  ADAPTIVE_MDAP_PES_INTEGRATION_DESIGN.md

For implementation details, see:
  adaptive_mdap_pes_integration.py
""")


if __name__ == "__main__":
    main()

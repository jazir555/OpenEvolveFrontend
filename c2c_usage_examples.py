"""
C2C MCP Tools - Usage Examples

This file demonstrates how to use the C2C (Cache-to-Cache) MCP tools
for multi-model ensemble inference with CrewAI integration.

EXAMPLES INCLUDED:
    1. Basic ensemble initialization and inference
    2. Team consensus for Decomposition workflow
    3. Hephaestus phase-specific configuration
    4. Ensemble cache management
    5. Error handling and graceful degradation
    6. Comparison with baseline models

Author: OpenEvolve C2C Integration
Version: 1.0.0
"""

import logging
from typing import Dict, Any, List

# Import C2C MCP tools
try:
    from c2c_mcp_tools import (
        initialize_c2c_ensemble,
        run_c2c_inference,
        run_team_consensus_with_c2c,
        configure_c2c_for_hephaestus_phase,
        get_c2c_status,
        load_c2c_checkpoint,
        compare_c2c_vs_baseline,
        manage_ensemble_cache,
        get_c2c_installation_guide,
        list_mcp_tools,
        C2C_AVAILABLE,
        C2CError,
        C2CNotAvailableError,
        C2CConfigurationError,
        C2CInferenceError,
        C2CCacheError,
    )
    print("[OK] C2C MCP tools imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import C2C tools: {e}")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Check C2C Status
# ============================================================================

def example_1_check_status():
    """Example 1: Check C2C installation and component status."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Check C2C Status")
    print("="*70)

    status = get_c2c_status()

    print(f"\nC2C Available: {status['available']}")
    if status['available']:
        print(f"Version: {status['version']}")
        print(f"CUDA Available: {status['cuda_available']}")
        if status['cuda_available']:
            print(f"CUDA Devices: {status['cuda_device_count']}")
            print(f"Device Name: {status['cuda_device_name']}")
        print(f"\nCache Statistics:")
        print(f"  Cached Ensembles: {status['cache_stats']['size']}/{status['cache_stats']['max_size']}")
        print(f"\nFeatures:")
        for feature, enabled in status['features'].items():
            print(f"  - {feature}: {enabled}")
        print(f"\nPerformance Benefits:")
        for benefit, value in status['performance_benefits'].items():
            print(f"  - {benefit}: {value}")
    else:
        print(f"Error: {status.get('error', 'Unknown error')}")
        print(f"\nInstallation Guide:")
        print(status.get('installation_guide', {}).get('install_command'))

    return status


# ============================================================================
# EXAMPLE 2: Initialize C2C Ensemble
# ============================================================================

def example_2_initialize_ensemble():
    """Example 2: Initialize a C2C ensemble with base and sharer models."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Initialize C2C Ensemble")
    print("="*70)

    if not C2C_AVAILABLE:
        print("\n⚠ C2C not available. This example requires C2C installation.")
        print("See installation guide in Example 1 output.")
        return None

    try:
        # Initialize ensemble with small models for demonstration
        result = initialize_c2c_ensemble(
            ensemble_id="demo-ensemble-1",
            base_model="Qwen/Qwen3-0.6B",
            sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
            device="cpu",  # Use "cuda" if GPU available
            include_response=True,
            multi_source_fusion_mode="parallel",
            cache_ensemble=True,
        )

        if result['success']:
            print(f"\n✓ Ensemble initialized successfully!")
            print(f"  Ensemble ID: {result['ensemble_id']}")
            print(f"  Number of Models: {result['num_models']}")
            print(f"  Base Model: {result['base_model']}")
            print(f"  Sharer Models: {result['sharer_models']}")
            print(f"  Device: {result['device']}")
            print(f"  Num Layers: {result['num_layers']}")
            print(f"  Cached: {result['cached']}")
        else:
            print(f"\n✗ Failed to initialize ensemble: {result.get('error')}")

        return result

    except C2CNotAvailableError as e:
        print(f"\n✗ C2C not available: {e}")
        return None
    except C2CConfigurationError as e:
        print(f"\n✗ Configuration error: {e}")
        return None
    except C2CError as e:
        print(f"\n✗ C2C error: {e}")
        return None


# ============================================================================
# EXAMPLE 3: Run C2C Inference
# ============================================================================

def example_3_run_inference(ensemble_id: str = "demo-ensemble-1"):
    """Example 3: Run inference using C2C ensemble."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Run C2C Inference")
    print("="*70)

    if not C2C_AVAILABLE:
        print("\n⚠ C2C not available. This example requires C2C installation.")
        return None

    try:
        # Run inference with ensemble
        result = run_c2c_inference(
            ensemble_id=ensemble_id,
            prompt="What is the capital of France? Explain briefly.",
            apply_c2c=True,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
        )

        if result['success']:
            print(f"\n✓ Inference completed successfully!")
            print(f"  Generated Text: {result['generated_text'][:200]}...")
            print(f"  Tokens Generated: {result['tokens_generated']}")
            print(f"  Inference Time: {result['inference_time']}s")
            print(f"  Tokens/Second: {result['tokens_per_second']}")
            print(f"  C2C Applied: {result['c2c_applied']}")
        else:
            print(f"\n✗ Inference failed: {result.get('error')}")

        return result

    except C2CCacheError as e:
        print(f"\n✗ Ensemble not found in cache: {e}")
        print("Hint: Initialize ensemble first using Example 2")
        return None
    except C2CInferenceError as e:
        print(f"\n✗ Inference error: {e}")
        return None
    except C2CError as e:
        print(f"\n✗ C2C error: {e}")
        return None


# ============================================================================
# EXAMPLE 4: Team Consensus with C2C
# ============================================================================

def example_4_team_consensus(ensemble_id: str = "demo-ensemble-1"):
    """Example 4: Run team consensus for Decomposition workflow."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Team Consensus for Decomposition Workflow")
    print("="*70)

    if not C2C_AVAILABLE:
        print("\n⚠ C2C not available. This example requires C2C installation.")
        return None

    try:
        # Simulate Blue Team consensus
        result = run_team_consensus_with_c2c(
            ensemble_id=ensemble_id,
            prompt="Design a solution for multi-model ensemble inference",
            team_name="Blue",
            team_models=["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"],
            consensus_mode="c2c",
            max_new_tokens=150,
        )

        if result['success']:
            print(f"\n✓ Team consensus completed!")
            print(f"  Team: {result['team_name']}")
            print(f"  Consensus Mode: {result['consensus_mode']}")
            print(f"  Team Members: {result['num_team_members']}")
            print(f"  Consensus Text: {result['consensus_text'][:300]}...")
        else:
            print(f"\n✗ Consensus failed: {result.get('error')}")

        return result

    except C2CConfigurationError as e:
        print(f"\n✗ Configuration error: {e}")
        return None
    except C2CError as e:
        print(f"\n✗ C2C error: {e}")
        return None


# ============================================================================
# EXAMPLE 5: Configure C2C for Hephaestus Phases
# ============================================================================

def example_5_hephaestus_configuration():
    """Example 5: Configure C2C for different Hephaestus phases."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Configure C2C for Hephaestus Phases")
    print("="*70)

    if not C2C_AVAILABLE:
        print("\n⚠ C2C not available. Showing recommendations only.")

    # Configure for different phases
    phases = ["setup", "solution", "critique", "verify", "reassemble", "final"]

    for phase in phases:
        print(f"\n--- Phase: {phase.upper()} ---")
        result = configure_c2c_for_hephaestus_phase(
            phase_id=f"hephaestus-{phase}",
            base_model="Qwen/Qwen3-0.6B",
            phase_type=phase,
            ensemble_config={"device": "cpu"},
        )

        if result['success']:
            recommendation = result.get('recommendation', {})
            print(f"  Description: {recommendation.get('description', 'N/A')}")
            print(f"  Recommended Pairs: {recommendation.get('recommended_pairs', [])}")
            print(f"  Reasoning: {recommendation.get('reasoning', 'N/A')}")
        else:
            print(f"  Error: {result.get('error')}")

    return result


# ============================================================================
# EXAMPLE 6: Ensemble Cache Management
# ============================================================================

def example_6_cache_management():
    """Example 6: Manage ensemble cache."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Ensemble Cache Management")
    print("="*70)

    try:
        # List cached ensembles
        print("\n1. List cached ensembles:")
        result = manage_ensemble_cache(action="list")
        if result['success']:
            print(f"   Cached ensembles: {result['cached_ensembles']}")
            print(f"   Total count: {result['count']}")

        # Get cache statistics
        print("\n2. Get cache statistics:")
        result = manage_ensemble_cache(action="stats")
        if result['success']:
            stats = result['stats']
            print(f"   Cache size: {stats['size']}/{stats['max_size']}")
            print(f"   Cached ensembles:")
            for ensemble in stats['cached_ensembles']:
                print(f"     - {ensemble['ensemble_id']}")
                print(f"       Created: {ensemble['created']}")
                print(f"       Last used: {ensemble['last_used']}")
                print(f"       Use count: {ensemble['use_count']}")

        # Configure persistent storage
        print("\n3. Configure persistent storage:")
        result = manage_ensemble_cache(
            action="config",
            persistent_path="./c2c_cache_metadata"
        )
        if result['success']:
            print(f"   ✓ Persistent storage: {result['persistent_path']}")

        return result

    except C2CCacheError as e:
        print(f"\n✗ Cache error: {e}")
        return None


# ============================================================================
# EXAMPLE 7: Compare C2C vs Baseline
# ============================================================================

def example_7_compare_baseline(ensemble_id: str = "demo-ensemble-1"):
    """Example 7: Compare C2C ensemble vs baseline model."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Compare C2C vs Baseline")
    print("="*70)

    if not C2C_AVAILABLE:
        print("\n⚠ C2C not available. Showing expected improvements based on research.")

    try:
        result = compare_c2c_vs_baseline(
            ensemble_id=ensemble_id,
            prompts=[
                "What is machine learning?",
                "Explain quantum computing.",
                "Describe neural networks."
            ],
            base_model="Qwen/Qwen3-0.6B",
            sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
            max_new_tokens=100,
        )

        if result['success']:
            print(f"\n✓ Comparison analysis ready!")
            print(f"  Ensemble ID: {result['ensemble_id']}")
            print(f"  Number of Prompts: {result['num_prompts']}")
            print(f"  Ensemble Loaded: {result['ensemble_loaded']}")
            print(f"\n  Expected Improvements:")
            for metric, value in result['expected_improvements'].items():
                if metric != 'source':
                    print(f"    - {metric}: {value}")
            if 'source' in result['expected_improvements']:
                print(f"\n  Source: {result['expected_improvements']['source']}")
        else:
            print(f"\n✗ Comparison failed: {result.get('error')}")

        return result

    except C2CError as e:
        print(f"\n✗ C2C error: {e}")
        return None


# ============================================================================
# EXAMPLE 8: Error Handling
# ============================================================================

def example_8_error_handling():
    """Example 8: Demonstrate error handling and graceful degradation."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Error Handling and Graceful Degradation")
    print("="*70)

    # Example 1: Handle missing ensemble
    print("\n1. Attempt inference with non-existent ensemble:")
    try:
        result = run_c2c_inference(
            ensemble_id="non-existent-ensemble",
            prompt="Test prompt",
        )
        print(f"   Result: {result}")
    except C2CCacheError as e:
        print(f"   ✓ Caught C2CCacheError: {e}")
    except C2CError as e:
        print(f"   ✓ Caught C2CError: {e}")

    # Example 2: Handle invalid device
    print("\n2. Attempt initialization with invalid device:")
    try:
        if C2C_AVAILABLE:
            result = initialize_c2c_ensemble(
                ensemble_id="test-invalid-device",
                base_model="Qwen/Qwen3-0.6B",
                sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
                device="invalid-device",
            )
            print(f"   Result: {result}")
    except C2CConfigurationError as e:
        print(f"   ✓ Caught C2CConfigurationError: {e}")
    except C2CError as e:
        print(f"   ✓ Caught C2CError: {e}")

    # Example 3: Check if C2C is available with graceful fallback
    print("\n3. Check C2C availability with fallback:")
    if C2C_AVAILABLE:
        print("   ✓ C2C is available - using C2C ensemble inference")
    else:
        print("   ⚠ C2C not available - falling back to single model inference")
        print("   Installation guide:")
        print(get_c2c_installation_guide())


# ============================================================================
# EXAMPLE 9: Integration with CrewAI
# ============================================================================

def example_9_crewai_integration():
    """Example 9: Integrate C2C tools with CrewAI agents."""
    print("\n" + "="*70)
    print("EXAMPLE 9: CrewAI Integration Pattern")
    print("="*70)

    print("""
# CrewAI Agent Integration Example:

from crewai import Agent, Task, Crew
from c2c_mcp_tools import initialize_c2c_ensemble, run_c2c_inference

# Step 1: Initialize C2C ensemble (done once at startup)
ensemble = initialize_c2c_ensemble(
    ensemble_id="crewai-ensemble",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    device="cuda",
    cache_ensemble=True,
)

# Step 2: Create CrewAI agent with C2C inference
researcher = Agent(
    role="Research Analyst",
    goal="Analyze complex topics using multi-model consensus",
    backstory="Expert researcher with access to C2C ensemble",
    tools=[c2c_inference_tool],
)

# Step 3: Define task
task = Task(
    description="Analyze the impact of C2C on multi-model systems",
    expected_output="Detailed analysis with consensus from multiple models",
    agent=researcher,
)

# Step 4: Create crew and execute
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()

# Custom C2C tool for CrewAI
def c2c_inference_tool(prompt: str) -> str:
    result = run_c2c_inference(
        ensemble_id="crewai-ensemble",
        prompt=prompt,
        apply_c2c=True,
        max_new_tokens=512,
    )
    return result['generated_text']
    """)


# ============================================================================
# EXAMPLE 10: Complete Workflow
# ============================================================================

def example_10_complete_workflow():
    """Example 10: Complete workflow from initialization to inference."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Complete Workflow")
    print("="*70)

    ensemble_id = "workflow-demo-ensemble"

    try:
        # Step 1: Check status
        print("\n1. Checking C2C status...")
        status = get_c2c_status()
        if not status['available']:
            print("   ⚠ C2C not available. Please install C2C first.")
            return

        # Step 2: Initialize ensemble
        print(f"\n2. Initializing ensemble '{ensemble_id}'...")
        init_result = initialize_c2c_ensemble(
            ensemble_id=ensemble_id,
            base_model="Qwen/Qwen3-0.6B",
            sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
            device="cpu",
            cache_ensemble=True,
        )

        if not init_result['success']:
            print(f"   ✗ Failed to initialize: {init_result.get('error')}")
            return

        print(f"   ✓ Ensemble initialized with {init_result['num_models']} models")

        # Step 3: Run inference
        print("\n3. Running inference...")
        inference_result = run_c2c_inference(
            ensemble_id=ensemble_id,
            prompt="What are the benefits of multi-model ensembles?",
            apply_c2c=True,
            max_new_tokens=150,
        )

        if inference_result['success']:
            print(f"   ✓ Generated {inference_result['tokens_generated']} tokens")
            print(f"   ✓ Speed: {inference_result['tokens_per_second']} tokens/s")
            print(f"\n   Generated Text:\n   {inference_result['generated_text']}")
        else:
            print(f"   ✗ Inference failed: {inference_result.get('error')}")

        # Step 4: Team consensus
        print("\n4. Running team consensus...")
        consensus_result = run_team_consensus_with_c2c(
            ensemble_id=ensemble_id,
            prompt="Evaluate the quality of multi-model consensus",
            team_name="Analysis",
            team_models=["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"],
            consensus_mode="c2c",
        )

        if consensus_result['success']:
            print(f"   ✓ Team consensus completed")
        else:
            print(f"   ✗ Consensus failed: {consensus_result.get('error')}")

        # Step 5: Check cache stats
        print("\n5. Checking cache statistics...")
        cache_stats = manage_ensemble_cache(action="stats")
        if cache_stats['success']:
            stats = cache_stats['stats']
            print(f"   Cache size: {stats['size']}/{stats['max_size']}")
            for ensemble in stats['cached_ensembles']:
                print(f"   - {ensemble['ensemble_id']}: {ensemble['use_count']} uses")

        print("\n✓ Complete workflow finished successfully!")

    except C2CError as e:
        print(f"\n✗ C2C error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

def main():
    """Run all examples to demonstrate C2C MCP tools functionality."""
    print("\n" + "="*70)
    print("C2C MCP TOOLS - USAGE EXAMPLES")
    print("="*70)
    print(f"\nC2C Available: {C2C_AVAILABLE}")
    print(f"C2C Version: 1.0.0")
    print(f"Number of MCP Tools: {len(list_mcp_tools())}")

    # Run examples
    example_1_check_status()
    example_2_initialize_ensemble()
    example_3_run_inference()
    example_4_team_consensus()
    example_5_hephaestus_configuration()
    example_6_cache_management()
    example_7_compare_baseline()
    example_8_error_handling()
    example_9_crewai_integration()
    example_10_complete_workflow()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70)
    print("\nNext Steps:")
    print("1. Install C2C (Rosetta) if not available")
    print("2. Initialize ensemble with your preferred models")
    print("3. Integrate with your CrewAI workflow")
    print("4. Monitor cache and performance")
    print("\nFor installation guide, use: get_c2c_installation_guide()")


if __name__ == "__main__":
    main()

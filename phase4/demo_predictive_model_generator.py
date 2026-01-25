"""
Delta-2 Predictive Model Generator - Demonstration
============================================

Demonstrates the complete predictive model generation pipeline.

Author: Agent E2 (Delta-2 Specialist)
Date: 2025-12-31
"""

import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
from phase4.predictive_model_generator import (
    PredictiveModelGenerator,
    RESESolution,
    Delta2Config,
    ModelType,
    generate_predictive_model,
)


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_usage():
    """Demonstrate basic model generation"""
    print_section("DEMO 1: Basic Predictive Model Generation")

    # Create RESE solution
    print("[Step 1] Creating RESE solution...")
    solution = RESESolution(
        problem_id="demo_001",
        solution={
            "optimization_result": {
                "objective": 0.92,
                "parameters": {"temp": 350, "pressure": 2.5}
            }
        },
        constraints=[
            "Temperature must be between 300-400 K",
            "Pressure must be > 1.0 bar",
            "Reaction time must be positive"
        ],
        aci_history=[65.0, 50.0, 35.0, 22.0, 15.0],
        metadata={
            "domain": "chemistry",
            "n_samples": 100
        }
    )
    print(f"  [OK] Problem: {solution.problem_id}")
    print(f"  [OK] Constraints: {len(solution.constraints)}")
    print(f"  [OK] ACI history: {solution.aci_history}")

    # Prepare training data
    print("\n[Step 2] Preparing training data...")
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100 samples, 3 features
    y = 2 * X[:, 0] + 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1
    print(f"  [OK] Samples: {X.shape[0]}")
    print(f"  [OK] Features: {X.shape[1]}")

    # Generate predictive model
    print("\n[Step 3] Generating predictive model...")
    model = generate_predictive_model(
        solution=solution,
        model_type=ModelType.RANDOM_FOREST,
        X=X,
        y=y
    )
    print(f"  [OK] Model type: {model.model_type.value}")
    print(f"  [OK] Architecture: {model.architecture}")
    print(f"  [OK] Features extracted: {len(model.features)}")

    # Display results
    print("\n[Step 4] Model Results:")
    print(f"  Falsifiable: {model.falsifiability.is_falsifiable}")
    print(f"  Testable predictions: {model.falsifiability.num_testable_predictions}")
    print(f"  R² Score: {model.metrics.r2_score:.3f}" if model.metrics.r2_score else "  N/A (classification)")
    print(f"  MSE: {model.metrics.mse:.3f}" if model.metrics.mse else "  N/A")

    # Display top features
    print("\n[Step 5] Top Features:")
    for i, feature in enumerate(model.features[:5], 1):
        print(f"  {i}. {feature.name}: {feature.importance:.3f}")

    # Display predictions
    print("\n[Step 6] Testable Predictions:")
    for i, pred in enumerate(model.predictions[:3], 1):
        print(f"  {i}. {pred.variable}: {pred.expected_value:.3f} (confidence: {pred.confidence:.2f})")

    print("\n[OK] Demo 1 Complete!")


def demo_interpretable_models():
    """Demonstrate interpretable model generation"""
    print_section("DEMO 2: Interpretable Decision Tree")

    # Create solution requiring interpretability
    print("[Step 1] Creating solution requiring interpretability...")
    solution = RESESolution(
        problem_id="medical_decision_001",
        solution={"diagnosis": "condition_A"},
        constraints=[
            "Age must be > 0",
            "Symptom severity: 0-10",
            "Lab results must be in normal range"
        ],
        metadata={
            "domain": "medicine",
            "require_interpretability": True
        }
    )
    print(f"  [OK] Domain: {solution.metadata['domain']}")
    print(f"  [OK] Requires interpretability: {solution.metadata['require_interpretability']}")

    # Configure for interpretable models
    print("\n[Step 2] Configuring for interpretable models...")
    config = Delta2Config(
        prefer_interpretable=True,
        tree_max_depth=5
    )
    print(f"  [OK] Prefer interpretable: {config.prefer_interpretable}")
    print(f"  [OK] Max depth: {config.tree_max_depth}")

    # Generate model
    print("\n[Step 3] Generating interpretable model...")
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary classification

    model = generate_predictive_model(
        solution=solution,
        model_type=ModelType.DECISION_TREE,
        config=config,
        X=X,
        y=y
    )

    print(f"  [OK] Model type: {model.model_type.value}")
    print(f"  [OK] Interpretable: Yes (Decision Tree)")
    print(f"  [OK] Accuracy: {model.metrics.accuracy:.3f}" if model.metrics.accuracy else "  N/A")

    print("\n[OK] Demo 2 Complete!")


def demo_uncertainty_quantification():
    """Demonstrate uncertainty quantification"""
    print_section("DEMO 3: Uncertainty Quantification")

    # Create solution
    print("[Step 1] Creating solution...")
    solution = RESESolution(
        problem_id="materials_design_002",
        solution={"material": "novel_composite"},
        constraints=[
            "Strength > 100 MPa",
            "Density < 5 g/cm³",
            "Cost < $100/kg"
        ],
        metadata={"domain": "materials_science"}
    )

    # Configure with uncertainty quantification
    print("\n[Step 2] Configuring uncertainty quantification...")
    config = Delta2Config(
        uncertainty_method="bootstrap",
        n_bootstrap_samples=50  # Small for demo
    )
    print(f"  [OK] Method: {config.uncertainty_method}")
    print(f"  [OK] Bootstrap samples: {config.n_bootstrap_samples}")

    # Generate model
    print("\n[Step 3] Generating model with uncertainty...")
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 200 + 50 * X[:, 0] + 30 * X[:, 1] + np.random.randn(100) * 5

    model = generate_predictive_model(
        solution=solution,
        model_type=ModelType.RANDOM_FOREST,
        config=config,
        X=X,
        y=y
    )

    print(f"  [OK] Model generated")
    print(f"  [OK] Uncertainty method: {model.uncertainty.method if model.uncertainty else 'N/A'}")

    if model.uncertainty:
        print(f"\n[Step 4] Confidence Intervals:")
        for feature_name, (lower, upper) in list(model.uncertainty.confidence_intervals.items())[:3]:
            print(f"  {feature_name}: [{lower:.3f}, {upper:.3f}]")

    print("\n[OK] Demo 3 Complete!")


def demo_delta1_integration():
    """Demonstrate integration with Delta-1 architecture"""
    print_section("DEMO 4: Delta-1 Architecture Integration")

    # Create solution with architecture
    print("[Step 1] Creating solution with Delta-1 architecture...")
    solution = RESESolution(
        problem_id="delta1_integration_demo",
        solution={},
        constraints=[
            "Module A output positive",
            "Module B processing time < 100ms",
            "Module C accuracy > 95%"
        ],
        architecture={
            "type": "pipeline",
            "components": [
                {"name": "preprocessing", "type": "transform"},
                {"name": "feature_extraction", "type": "extract"},
                {"name": "prediction", "type": "predict"}
            ]
        },
        metadata={
            "domain": "machine_learning",
            "architecture_source": "delta1"
        }
    )
    print(f"  [OK] Architecture type: {solution.architecture['type']}")
    print(f"  [OK] Components: {len(solution.architecture['components'])}")

    # Generate model
    print("\n[Step 2] Generating model from architecture...")
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 3 modules
    y = np.random.randn(100)

    model = generate_predictive_model(
        solution=solution,
        model_type=ModelType.DECISION_TREE,
        X=X,
        y=y
    )

    print(f"  [OK] Model generated")
    print(f"  [OK] Features: {len(model.features)}")
    print(f"  [OK] Architecture source: {model.metadata.get('solution_metadata', {}).get('architecture_source', 'N/A')}")

    print("\n[OK] Demo 4 Complete!")


def demo_stage8_integration():
    """Demonstrate Stage 8 E2E integration"""
    print_section("DEMO 5: Stage 8 E2E Integration")

    # Create complete RESE solution
    print("[Step 1] Creating complete RESE solution...")
    solution = RESESolution(
        problem_id="stage8_demo",
        solution={
            "invention": "novel_synthesis_method",
            "parameters": {
                "temperature": 450,
                "catalyst": "Cu-Zn",
                "time": 2.5
            }
        },
        constraints=[
            "Temperature < 500°C",
            "Catalyst concentration > 0.1 M",
            "Time > 1 hour"
        ],
        aci_history=[70.0, 55.0, 38.0, 25.0, 18.0, 12.0],
        metadata={
            "domain": "chemistry",
            "target_stage": 8
        },
        stage_results={
            "stage1": {"status": "complete"},
            "stage2": {"status": "complete"},
            "stage3": {"status": "complete"},
            "stage4": {"status": "complete"},
            "stage5": {"status": "complete"},
            "stage6": {"status": "complete"},
            "stage7": {"status": "complete"}
        }
    )
    print(f"  [OK] Problem: {solution.problem_id}")
    print(f"  [OK] Stages complete: {len(solution.stage_results)}")

    # Generate Stage 8 ready model
    print("\n[Step 2] Generating Stage 8 ready model...")
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)

    model = generate_predictive_model(
        solution=solution,
        model_type=ModelType.RANDOM_FOREST,
        X=X,
        y=y
    )

    print(f"  [OK] Model generated")
    print(f"  [OK] Falsifiable: {model.falsifiability.is_falsifiable}")
    print(f"  [OK] Testable predictions: {model.falsifiability.num_testable_predictions}")

    # Display Stage 8 outputs
    print("\n[Step 3] Stage 8 Outputs:")
    print(f"  Predictive Model: {model.architecture}")
    print(f"  Predictions: {len(model.predictions)} testable predictions")
    print(f"  Validation Metrics: R² = {model.metrics.r2_score:.3f}" if model.metrics.r2_score else "  Accuracy = {:.3f}".format(model.metrics.accuracy))
    print(f"  Falsifiability Report: {'PASS' if model.falsifiability.is_falsifiable else 'FAIL'}")

    print("\n[Step 4] Standard Operating Procedure (SOP) Preview:")
    print("  1. Set temperature to specified range")
    print("  2. Add catalyst at recommended concentration")
    print("  3. Monitor reaction for specified time")
    print("  4. Validate predictions against experimental results")
    print("  5. Iterate based on prediction confidence")

    print("\n[OK] Demo 5 Complete!")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("  Delta-2 PREDICTIVE MODEL GENERATOR - DEMONSTRATION")
    print("  Agent E2 (Delta-2 Specialist)")
    print("  Date: 2025-12-31")
    print("=" * 70)

    try:
        # Run demos
        demo_basic_usage()
        demo_interpretable_models()
        demo_uncertainty_quantification()
        demo_delta1_integration()
        demo_stage8_integration()

        # Summary
        print_section("DEMONSTRATION COMPLETE")
        print("[OK] All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  1. Basic predictive model generation")
        print("  2. Interpretable decision tree models")
        print("  3. Uncertainty quantification (bootstrap)")
        print("  4. Delta-1 Architecture Assembly integration")
        print("  5. Stage 8 E2E pipeline integration")
        print("\nFor more information, see:")
        print("  - rese/docs/predictive_models_research.md (Research)")
        print("  - rese/docs/AGENT_E2_COMPLETION_REPORT.md (Full Report)")
        print("  - rese/phase4/predictive_model_generator.py (Implementation)")
        print("\nDelta-2 Predictive Model Generator - Mission Accomplished!")

    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

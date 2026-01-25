"""
Example 11: Error Handling and Debugging

This example demonstrates how to handle errors and debug RESE pipelines.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

import logging
from rese_pipeline import RESEPipeline, ProblemInput
from core.symbolic_constraint_engine import Constraint, ConstraintType
from config import RESEConfig

def setup_logging():
    """Configure logging for debugging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rese_debug.log'),
            logging.StreamHandler()
        ]
    )

def example_basic_error_handling():
    """Example 1: Basic error handling"""
    print("=" * 60)
    print("Example 1: Basic Error Handling")
    print("=" * 60)
    print()

    try:
        # This will fail - missing required fields
        from rese_pipeline import ProblemInput

        problem = ProblemInput(
            id="test",
            description="Test",  # Missing constraints
            # variables=[]       # Missing variables
        )

    except TypeError as e:
        print(f"✓ Caught TypeError: {e}")
        print("  This is expected - ProblemInput requires all fields")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print()

def example_constraint_validation():
    """Example 2: Constraint validation"""
    print("=" * 60)
    print("Example 2: Constraint Validation")
    print("=" * 60)
    print()

    from core.symbolic_constraint_engine import SymbolicConstraintEngine

    sce = SymbolicConstraintEngine()

    # Try to add invalid constraint
    try:
        invalid_constraint = Constraint(
            id="",  # Empty ID - will fail
            type=ConstraintType.HARD,
            description="Test",
            formalization="x > 0",
            source="test"
        )
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
        print("  Constraint validation working correctly")

    print()

    # Add valid constraint and check conflicts
    c1 = Constraint(
        id="c1",
        type=ConstraintType.HARD,
        description="x > 0",
        formalization="x > 0",
        source="test"
    )

    c2 = Constraint(
        id="c2",
        type=ConstraintType.HARD,
        description="x < 0",
        formalization="x < 0",
        source="test"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)

    conflicts = sce.detect_conflicts()
    print(f"Detected {len(conflicts)} conflicts:")
    for c1_id, c2_id in conflicts:
        print(f"  - {c1_id} vs {c2_id}")

    print()

def example_pipeline_error_recovery():
    """Example 3: Pipeline error recovery"""
    print("=" * 60)
    print("Example 3: Pipeline Error Recovery")
    print("=" * 60)
    print()

    from rese_pipeline import RESEPipeline, ProblemInput
    from config import PipelineConfig

    # Create problem with potentially problematic constraints
    problem = ProblemInput(
        id="error_test",
        description="Test error handling",
        constraints=[
            {
                "id": "c1",
                "type": "hard",
                "description": "Valid constraint",
                "formalization": "x > 0"
            }
        ],
        variables={"x": 10}
    )

    # Configure pipeline with error handling
    config = RESEConfig(
        pipeline=PipelineConfig(
            max_retries=2,
            continue_on_error=False,
            rollback_on_failure=True
        )
    )

    pipeline = RESEPipeline(config)

    # Add progress callback to monitor execution
    def monitor_progress(result):
        print(f"Phase Status: {result.status.value}")
        if result.errors:
            print(f"  Errors: {result.errors[:2]}")  # Show first 2

    pipeline.add_progress_callback(monitor_progress)

    try:
        result = pipeline.run(problem)
        print(f"✓ Pipeline completed: {result.status.value}")
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        print("  This demonstrates error handling in pipelines")

    print()

def example_debugging_with_inspection():
    """Example 4: Debugging with inspection"""
    print("=" * 60)
    print("Example 4: Debugging with Inspection")
    print("=" * 60)
    print()

    from core.symbolic_constraint_engine import SymbolicConstraintEngine

    sce = SymbolicConstraintEngine()

    # Add constraints
    constraints = [
        Constraint(
            id=f"c{i}",
            type=ConstraintType.HARD if i % 2 == 0 else ConstraintType.SOFT,
            description=f"Constraint {i}",
            formalization=f"x_{i} > 0",
            source="test"
        )
        for i in range(5)
    ]

    for constraint in constraints:
        sce.add_constraint(constraint)

    # Inspect engine state
    print("Engine State Inspection:")
    print(f"  Total constraints: {len(sce.get_all_constraints())}")
    print(f"  Hard constraints: {sum(1 for c in sce.get_all_constraints() if c.is_hard())}")
    print(f"  Soft constraints: {sum(1 for c in sce.get_all_constraints() if not c.is_hard())}")

    # Get execution order
    order = sce.get_execution_order()
    print(f"  Execution order: {' → '.join(order)}")

    # Validate
    validation = sce.validate()
    print(f"  Valid: {validation['is_valid']}")

    print()

def example_performance_profiling():
    """Example 5: Performance profiling"""
    print("=" * 60)
    print("Example 5: Performance Profiling")
    print("=" * 60)
    print()

    import time
    from core.symbolic_constraint_engine import SymbolicConstraintEngine

    # Create large constraint set
    num_constraints = 1000

    print(f"Creating {num_constraints} constraints...")
    start = time.time()

    sce = SymbolicConstraintEngine()

    for i in range(num_constraints):
        constraint = Constraint(
            id=f"c{i}",
            type=ConstraintType.HARD if i % 3 == 0 else ConstraintType.SOFT,
            description=f"Constraint {i}",
            formalization=f"x_{i} > {i % 10}",
            source="test"
        )
        sce.add_constraint(constraint)

    creation_time = time.time() - start
    print(f"  Time: {creation_time:.4f}s")
    print()

    # Profile conflict detection
    print("Detecting conflicts...")
    start = time.time()

    conflicts = sce.detect_conflicts()

    detection_time = time.time() - start
    print(f"  Time: {detection_time:.4f}s")
    print(f"  Conflicts found: {len(conflicts)}")

    # Calculate rate
    print(f"  Rate: {num_constraints/detection_time:.0f} constraints/second")

    print()

def example_logging_setup():
    """Example 6: Logging setup"""
    print("=" * 60)
    print("Example 6: Logging Setup")
    print("=" * 60)
    print()

    # Setup logging
    setup_logging()

    logger = logging.getLogger('rese')

    print("Logging configured - check 'rese_debug.log' for output")
    print()

    # Log various levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    print("Messages logged. Check rese_debug.log file.")

    print()

def example_common_pitfalls():
    """Example 7: Common pitfalls and solutions"""
    print("=" * 60)
    print("Example 7: Common Pitfalls and Solutions")
    print("=" * 60)
    print()

    pitfalls = [
        {
            'pitfall': 'Forgetting to add path to sys.path',
            'solution': 'sys.path.insert(0, "/path/to/rese")',
            'example': 'See top of this file'
        },
        {
            'pitfall': 'Using wrong constraint type',
            'solution': 'Use ConstraintType.HARD, SOFT, or PREFERENCE',
            'example': 'Constraint(id="c1", type=ConstraintType.HARD, ...)'
        },
        {
            'pitfall': 'Not checking pipeline status',
            'solution': 'Always check result.status before using output',
            'example': 'if result.status.value == "completed": ...'
        },
        {
            'pitfall': 'Ignoring phase warnings',
            'solution': 'Review phase_result.warnings after execution',
            'example': 'for warning in phase_result.warnings: print(warning)'
        },
        {
            'pitfall': 'Not enabling caching',
            'solution': 'Set enable_caching=True in PipelineConfig',
            'example': 'config = RESEConfig(pipeline=PipelineConfig(enable_caching=True))'
        }
    ]

    for i, item in enumerate(pitfalls, 1):
        print(f"{i}. Pitfall: {item['pitfall']}")
        print(f"   Solution: {item['solution']}")
        print(f"   Example: {item['example']}")
        print()

def main():
    print("=" * 70)
    print("Example 11: Error Handling and Debugging")
    print("=" * 70)
    print()

    # Run all examples
    example_basic_error_handling()
    example_constraint_validation()
    example_pipeline_error_recovery()
    example_debugging_with_inspection()
    example_performance_profiling()
    example_logging_setup()
    example_common_pitfalls()

    print("=" * 70)
    print("Summary: Error Handling Best Practices")
    print("=" * 70)
    print()
    print("1. ✓ Always validate inputs before processing")
    print("2. ✓ Use try-except blocks for error-prone operations")
    print("3. ✓ Configure logging for debugging")
    print("4. ✓ Monitor pipeline progress with callbacks")
    print("5. ✓ Profile performance to identify bottlenecks")
    print("6. ✓ Review warnings and errors after execution")
    print("7. ✓ Use caching to avoid redundant computations")
    print()

    print("=" * 70)
    print("Example 11 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

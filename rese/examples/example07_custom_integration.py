"""
Example 7: Custom Phase Integration

This example demonstrates how to create a custom phase and integrate it
into the RESE pipeline.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from rese_pipeline import PhaseExecutor, PhaseResult, PhaseStatus
from config import RESEConfig
from datetime import datetime
from typing import Any

class CustomPhaseExecutor(PhaseExecutor):
    """Example custom phase: Simple heuristic solver"""

    def __init__(self, phase_name: str, config: RESEConfig):
        super().__init__(phase_name, config)

    def execute(self, input_data: Any) -> PhaseResult:
        """Execute custom phase"""
        start_time = datetime.now()
        result = PhaseResult(
            phase_name=self.phase_name,
            status=PhaseStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Extract problem data
            constraints = input_data.get('constraints', [])
            variables = input_data.get('variables', {})

            # Apply simple heuristic
            # (In real usage, this would be domain-specific logic)
            solution = self._solve_heuristic(constraints, variables)

            result.output = {
                'solution': solution,
                'method': 'custom_heuristic',
                'iterations': len(constraints) * 10
            }

            result.metrics = {
                'constraints_processed': len(constraints),
                'variables_processed': len(variables),
                'solution_quality': 0.85
            }

            result.status = PhaseStatus.COMPLETED

        except Exception as e:
            result.status = PhaseStatus.FAILED
            result.errors.append(str(e))

        result.end_time = datetime.now()
        result.elapsed_seconds = (result.end_time - start_time).total_seconds()

        return result

    def _solve_heuristic(self, constraints, variables):
        """Simple heuristic solver"""
        # Placeholder: Greedy algorithm
        return {
            'values': [variables.get(k, 0) for k in variables.keys()],
            'objective': 42.0,
            'constraints_satisfied': len(constraints)
        }

def main():
    print("=" * 60)
    print("Example 7: Custom Phase Integration")
    print("=" * 60)
    print()

    # Create custom phase executor
    print("Creating Custom Phase:")
    print("-" * 60)

    config = RESEConfig()
    custom_executor = CustomPhaseExecutor("custom_heuristic", config)

    print(f"Phase Name: {custom_executor.phase_name}")
    print(f"Status: Ready")
    print()

    # Prepare input data
    input_data = {
        'constraints': [
            {'id': 'c1', 'type': 'hard', 'description': 'x > 0'},
            {'id': 'c2', 'type': 'soft', 'description': 'minimize x'}
        ],
        'variables': {
            'x': 10,
            'y': 20,
            'z': 30
        }
    }

    print("Input Data:")
    print(f"  Constraints: {len(input_data['constraints'])}")
    print(f"  Variables: {len(input_data['variables'])}")
    print()

    # Execute custom phase
    print("Executing Custom Phase:")
    print("-" * 60)

    result = custom_executor.execute(input_data)

    print()
    print("Execution Results:")
    print("-" * 60)
    print(f"Phase Name: {result.phase_name}")
    print(f"Status: {result.status.value}")
    print(f"Elapsed Time: {result.elapsed_seconds:.4f}s")
    print()

    if result.status == PhaseStatus.COMPLETED:
        print("Output:")
        for key, value in result.output.items():
            print(f"  {key}: {value}")
        print()

        print("Metrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
    else:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

    print()

    # Integration example
    print("=" * 60)
    print("Integration with RESE Pipeline:")
    print("-" * 60)
    print()
    print("To integrate a custom phase:")
    print()
    print("1. Create a custom executor class (as shown above)")
    print("2. Add it to RESEPipeline:")
    print()
    print("   from rese_pipeline import RESEPipeline")
    print("   pipeline = RESEPipeline()")
    print("   pipeline.custom_executor = custom_executor")
    print()
    print("3. Execute it alongside other phases:")
    print()
    print("   result = pipeline.run(")
    print("       problem,")
    print("       phases=['phase1', 'custom_heuristic', 'phase3']")
    print("   )")
    print()

    print("=" * 60)
    print("Example 7 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

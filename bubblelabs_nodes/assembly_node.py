"""
Assembly Node for BubbleLabs Integration

Implements solution merging and assembly from multiple sources.
"""

from typing import Dict, Any, List, Optional
from time import time
from .base_node import BubbleLabsNode, NodeExecutionError


class AssemblyNode(BubbleLabsNode):
    """
    Merges multiple solutions into a cohesive final solution.

    Supports various merge strategies:
    - weighted: Weighted combination of solutions
    - voting: Democratic voting on solution elements
    - expert_selection: Select best solution from experts
    - custom: User-defined merge logic
    """

    # Node metadata
    DISPLAY_NAME = "Solution Assembly"
    DESCRIPTION = (
        "Merge and assemble multiple solutions into a cohesive "
        "final solution with conflict resolution."
    )
    ICON = "assembly"
    CATEGORY = "integration"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import assembler (safe import)
        SolutionAssembler = self.safe_import(
            'solution_assembly.SolutionAssembler',
            fallback_value=None,
            error_msg="SolutionAssembler not available for AssemblyNode"
        )

        if SolutionAssembler:
            try:
                self.assembler = SolutionAssembler()
            except Exception as e:
                self.logger.warning(f"Could not instantiate SolutionAssembler: {e}")
                self.assembler = None
        else:
            self.assembler = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - solutions: List[Dict] containing multiple solutions

        Optional:
            - merge_strategy: str (weighted, voting, expert_selection, custom)
            - conflict_resolution: str (automatic, manual, hybrid)
            - quality_weights: Dict[str, float]
        """
        errors = []

        # Check required fields
        if 'solutions' not in inputs:
            errors.append("Missing required field: solutions")
        elif not isinstance(inputs['solutions'], list):
            errors.append("solutions must be a list")
        elif len(inputs['solutions']) == 0:
            errors.append("solutions list cannot be empty")
        elif len(inputs['solutions']) < 2:
            errors.append("At least 2 solutions required for assembly")

        # Validate merge_strategy
        if 'merge_strategy' in inputs:
            valid_strategies = ['weighted', 'voting', 'expert_selection', 'custom']
            if inputs['merge_strategy'] not in valid_strategies:
                errors.append(f"merge_strategy must be one of: {', '.join(valid_strategies)}")

        # Validate conflict_resolution
        if 'conflict_resolution' in inputs:
            valid_resolutions = ['automatic', 'manual', 'hybrid']
            if inputs['conflict_resolution'] not in valid_resolutions:
                errors.append(f"conflict_resolution must be one of: {', '.join(valid_resolutions)}")

        # Validate quality_weights
        if 'quality_weights' in inputs and inputs['quality_weights'] is not None:
            if not isinstance(inputs['quality_weights'], dict):
                errors.append("quality_weights must be a dictionary")
            else:
                for key, value in inputs['quality_weights'].items():
                    if not isinstance(value, (int, float)):
                        errors.append(f"quality_weights['{key}'] must be a number")
                    elif value < 0 or value > 1:
                        errors.append(f"quality_weights['{key}'] must be between 0 and 1")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Assemble multiple solutions into a final solution.

        Args:
            inputs: Must contain 'solutions' list and optional merge parameters
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - assembled_solution: The merged solution
                - conflicts_resolved: Number of conflicts resolved
                - quality_score: Overall quality score (0-1)
                - merge_report: Detailed merge report
                - discarded_elements: List of discarded solution elements
                - icr_statistics: ICR-related statistics (if enabled)
        """
        start_time = time()
        if not self.assembler:
            return self._assemble_simple(inputs, context)

        solutions = inputs['solutions']
        merge_strategy = inputs.get('merge_strategy', self.config.get('merge_strategy', 'weighted'))
        conflict_resolution = inputs.get('conflict_resolution', self.config.get('conflict_resolution', 'automatic'))
        quality_weights = inputs.get('quality_weights', self.config.get('quality_weights'))

        # Update progress
        context.update_progress(10, "Initializing solution assembler")
        self.logger.info(f"Assembling {len(solutions)} solutions using {merge_strategy} strategy")

        try:
            # Validate solutions
            context.update_progress(20, "Validating input solutions")

            for i, solution in enumerate(solutions):
                if not isinstance(solution, dict):
                    raise ValueError(f"Solution {i} is not a valid dictionary")

            # Perform assembly
            context.update_progress(30, "Analyzing solution compatibility")

            assembly_result = self.assembler.assemble(
                solutions=solutions,
                strategy=merge_strategy,
                conflict_resolution=conflict_resolution,
                quality_weights=quality_weights,
                callback=lambda p, m: context.update_progress(30 + p * 0.6, m)
            )

            # Update progress
            context.update_progress(90, "Finalizing assembled solution")

            # Extract and format results
            result = {
                'assembled_solution': assembly_result.final_solution,
                'conflicts_resolved': assembly_result.conflicts_resolved,
                'conflicts_detected': assembly_result.conflicts_detected,
                'quality_score': assembly_result.quality_score,
                'merge_report': {
                    'strategy_used': merge_strategy,
                    'conflict_resolution': conflict_resolution,
                    'solutions_count': len(solutions),
                    'elements_merged': assembly_result.elements_merged,
                    'elements_discarded': assembly_result.elements_discarded,
                    'merge_details': assembly_result.details
                },
                'discarded_elements': assembly_result.discarded_elements,
                'solution_sources': [s.get('source', f'solution_{i}') for i, s in enumerate(solutions)]
            }

            # Add artifacts to context
            context.add_artifact('assembly', {
                'result': result,
                'input_solutions': solutions,
                'strategy': merge_strategy
            })

            context.update_progress(
                100,
                f"Assembly complete: {result['conflicts_resolved']} conflicts resolved, "
                f"quality score: {result['quality_score']:.2f}"
            )

            self.logger.info(
                f"Solution assembly completed: {result['conflicts_resolved']}/{result['conflicts_detected']} "
                f"conflicts resolved, quality: {result['quality_score']:.2f}"
            )

            # ICR: Store assembly pattern
            execution_time = time() - start_time
            self.store_icr_pattern(
                operation_type='assembly',
                success=True,
                execution_time=execution_time,
                metadata={
                    'merge_strategy': merge_strategy,
                    'conflict_resolution': conflict_resolution,
                    'solutions_count': len(solutions),
                    'conflicts_resolved': result['conflicts_resolved'],
                    'conflicts_detected': result['conflicts_detected'],
                    'quality_score': result['quality_score']
                },
                sub_key=merge_strategy
            )

            # Include ICR statistics in result
            if self.enable_icr:
                result['icr_statistics'] = self.get_icr_statistics()

            return result

        except Exception as e:
            execution_time = time() - start_time
            # ICR: Store failed assembly pattern
            self.store_icr_pattern(
                operation_type='assembly',
                success=False,
                execution_time=execution_time,
                metadata={
                    'merge_strategy': merge_strategy,
                    'conflict_resolution': conflict_resolution,
                    'solutions_count': len(solutions),
                    'error': str(e),
                    'exception_type': type(e).__name__
                },
                sub_key=merge_strategy
            )
            self.logger.error(f"Solution assembly failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Solution assembly failed: {str(e)}",
                details={
                    'solutions_count': len(solutions),
                    'merge_strategy': merge_strategy,
                    'conflict_resolution': conflict_resolution,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _assemble_simple(self, inputs: Dict, context) -> Dict[str, Any]:
        """Simple assembly fallback when assembler not available"""
        start_time = time()
        solutions = inputs['solutions']
        merge_strategy = inputs.get('merge_strategy', 'weighted')

        context.update_progress(10, "Using simple assembly (assembler not available)")

        # Simple merge: combine all solutions into one structure
        assembled = {
            'merged_solutions': solutions,
            'merge_strategy': merge_strategy,
            'solutions_count': len(solutions),
            'note': 'Full assembler not available, using simple combination'
        }

        # Detect basic conflicts (same keys with different values)
        conflicts = []
        all_keys = set()
        for sol in solutions:
            all_keys.update(sol.keys())

        for key in all_keys:
            values = [sol.get(key) for sol in solutions if key in sol]
            if len(set(str(v) for v in values)) > 1:
                conflicts.append({
                    'key': key,
                    'conflicting_values': values
                })

        result = {
            'assembled_solution': assembled,
            'conflicts_resolved': 0,
            'conflicts_detected': len(conflicts),
            'quality_score': 0.5,
            'merge_report': {
                'strategy_used': merge_strategy,
                'conflict_resolution': 'none',
                'solutions_count': len(solutions),
                'elements_merged': len(all_keys),
                'elements_discarded': 0,
                'conflicts': conflicts,
                'merge_details': 'Simple merge performed'
            },
            'discarded_elements': [],
            'solution_sources': [s.get('source', f'solution_{i}') for i, s in enumerate(solutions)]
        }

        context.update_progress(100, "Simple assembly complete")

        # ICR: Store simple assembly pattern
        execution_time = time() - start_time
        self.store_icr_pattern(
            operation_type='assembly',
            success=True,
            execution_time=execution_time,
            metadata={
                'merge_strategy': merge_strategy,
                'solutions_count': len(solutions),
                'conflicts_detected': len(conflicts),
                'quality_score': 0.5,
                'is_simple_assembly': True
            },
            sub_key=merge_strategy
        )

        # Include ICR statistics in result
        if self.enable_icr:
            result['icr_statistics'] = self.get_icr_statistics()

        return result

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters"""
        return {
            "type": "object",
            "title": "Assembly Configuration",
            "description": "Configure solution merging and assembly parameters",
            "properties": {
                "merge_strategy": {
                    "type": "string",
                    "title": "Merge Strategy",
                    "description": "Strategy for combining multiple solutions",
                    "enum": ["weighted", "voting", "expert_selection", "custom"],
                    "enumNames": [
                        "Weighted Combination",
                        "Democratic Voting",
                        "Expert Selection",
                        "Custom Logic"
                    ],
                    "default": "weighted"
                },
                "conflict_resolution": {
                    "type": "string",
                    "title": "Conflict Resolution",
                    "description": "How to handle conflicts between solutions",
                    "enum": ["automatic", "manual", "hybrid"],
                    "enumNames": [
                        "Automatic Resolution",
                        "Manual Review",
                        "Hybrid Approach"
                    ],
                    "default": "automatic"
                },
                "quality_weights": {
                    "type": "object",
                    "title": "Quality Weights",
                    "description": "Weights for different quality metrics (0-1)",
                    "properties": {
                        "completeness": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.3
                        },
                        "correctness": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.4
                        },
                        "clarity": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.2
                        },
                        "efficiency": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.1
                        }
                    }
                }
            },
            "required": ["merge_strategy", "conflict_resolution"]
        }

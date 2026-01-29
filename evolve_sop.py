#!/usr/bin/env python3
"""
SOP Evolution Script - Automated Ensemble-Based SOP Improvement

Usage:
    python evolve_sop.py --input SOP.txt --output SOP_v16.2.txt --iterations 3

This script uses the integrated Red Team / Blue Team / Evaluator ensemble
to systematically identify vulnerabilities, generate fixes, and validate
improvements to technical Standard Operating Procedures.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import ensemble components
from red_team import RedTeam
from blue_team import BlueTeam, BlueTeamStrategy
from evaluator_team import EvaluatorTeam
from integrated_workflow import run_ensemble_based_workflow


class SOPEvolver:
    """Automated SOP evolution using ensemble methods"""

    def __init__(
        self,
        api_key: str,
        red_team_models: List[str],
        blue_team_models: List[str],
        evaluator_models: List[str],
        num_ensemble_models: int = 7
    ):
        self.api_key = api_key
        self.red_team_models = red_team_models
        self.blue_team_models = blue_team_models
        self.evaluator_models = evaluator_models
        self.num_ensemble_models = num_ensemble_models

        # Initialize teams
        self.red_team = RedTeam()
        self.blue_team = BlueTeam()
        self.evaluator_team = EvaluatorTeam()

    def evolve_sop(
        self,
        input_sop_path: str,
        output_sop_path: str,
        max_iterations: int = 3,
        quality_threshold: float = 0.90,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Evolve SOP through ensemble-driven workflow

        Args:
            input_sop_path: Path to original SOP
            output_sop_path: Path to save evolved SOP
            max_iterations: Maximum red/blue/eval cycles
            quality_threshold: Target quality score (0-1)
            save_intermediate: Save intermediate versions

        Returns:
            Dict containing evolution results
        """
        print(f"\n{'='*70}")
        print(f"SOP EVOLUTION: {input_sop_path}")
        print(f"{'='*70}\n")

        # Read original SOP
        with open(input_sop_path, 'r', encoding='utf-8') as f:
            original_sop = f.read()

        print(f"Original SOP length: {len(original_sop)} characters")
        print(f"Target quality threshold: {quality_threshold}")
        print(f"Max iterations: {max_iterations}\n")

        # Store evolution history
        evolution_history = {
            "original_version": input_sop_path,
            "final_version": output_sop_path,
            "timestamp": datetime.now().isoformat(),
            "iterations": [],
            "ensemble_config": {
                "red_team_models": self.red_team_models,
                "blue_team_models": self.blue_team_models,
                "evaluator_models": self.evaluator_models,
                "num_ensemble_models": self.num_ensemble_models
            }
        }

        current_sop = original_sop

        # Run evolution iterations
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'─'*70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'─'*70}\n")

            # Phase 1: Red Team Analysis
            print("Phase 1: Red Team Vulnerability Analysis...")
            red_team_result = self._run_red_team_analysis(current_sop)
            vulnerabilities_found = len(red_team_result.vulnerabilities)
            print(f"  ✓ Found {vulnerabilities_found} vulnerabilities")

            if vulnerabilities_found == 0:
                print("  No vulnerabilities found - SOP is robust!")
                break

            # Phase 2: Blue Team Fixes
            print("\nPhase 2: Blue Team Fix Generation...")
            blue_team_result = self._run_blue_team_fixes(
                current_sop,
                red_team_result
            )
            fixes_applied = len(blue_team_result.applied_fixes)
            print(f"  ✓ Applied {fixes_applied} fixes")
            current_sop = blue_team_result.fixed_content

            # Phase 3: Evaluator Assessment
            print("\nPhase 3: Evaluator Team Quality Assessment...")
            eval_result = self._run_evaluation(current_sop)
            quality_score = eval_result.consensus_score
            print(f"  ✓ Quality score: {quality_score:.3f}")

            # Record iteration results
            iteration_record = {
                "iteration": iteration,
                "vulnerabilities_found": vulnerabilities_found,
                "fixes_applied": fixes_applied,
                "quality_score": quality_score,
                "passed": quality_score >= quality_threshold
            }
            evolution_history["iterations"].append(iteration_record)

            # Save intermediate version if requested
            if save_intermediate:
                intermediate_path = f"{output_sop_path}.iter{iteration}"
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    f.write(current_sop)
                print(f"  Saved intermediate: {intermediate_path}")

            # Check if threshold met
            if quality_score >= quality_threshold:
                print(f"\n✓ QUALITY THRESHOLD MET ({quality_score:.3f} >= {quality_threshold})")
                break

            print(f"  Quality score {quality_score:.3f} below threshold {quality_threshold}")
            print(f"  Continuing to iteration {iteration + 1}...")

        # Save final evolved SOP
        with open(output_sop_path, 'w', encoding='utf-8') as f:
            f.write(current_sop)

        # Generate summary
        print(f"\n{'='*70}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*70}\n")

        print(f"Original → Final:")
        print(f"  Input:  {input_sop_path}")
        print(f"  Output: {output_sop_path}")
        print(f"\nIterations: {len(evolution_history['iterations'])}")
        print(f"Final quality score: {evolution_history['iterations'][-1]['quality_score']:.3f}")

        total_vulnerabilities = sum(it['vulnerabilities_found'] for it in evolution_history['iterations'])
        total_fixes = sum(it['fixes_applied'] for it in evolution_history['iterations'])
        print(f"Total vulnerabilities found: {total_vulnerabilities}")
        print(f"Total fixes applied: {total_fixes}")

        # Save evolution metadata
        metadata_path = output_sop_path + ".metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_history, f, indent=2)

        print(f"\nEvolution metadata saved: {metadata_path}")

        return evolution_history

    def _run_red_team_analysis(self, sop_content: str):
        """Run red team analysis on SOP"""
        return self.red_team.analyze_with_ensemble(
            content=sop_content,
            content_type="technical_sop",
            api_key=self.api_key,
            model_name=self.red_team_models[0],
            num_models=self.num_ensemble_models,
            attack_types=[
                "unrealistic_tolerance",
                "missing_contingency",
                "contradictory_requirement",
                "safety_vulnerability",
                "scaling_limitation",
                "measurement_uncertainty",
                "timing_conflict",
                "human_error_prone"
            ]
        )

    def _run_blue_team_fixes(self, sop_content: str, red_team_result):
        """Run blue team fix generation"""
        # Convert red team vulnerabilities to IssueFinding format
        from red_team import IssueFinding, IssueCategory, SeverityLevel

        issues = []
        for vuln in red_team_result.vulnerabilities[:10]:  # Limit to top 10
            severity_map = {
                'critical': SeverityLevel.CRITICAL,
                'high': SeverityLevel.HIGH,
                'medium': SeverityLevel.MEDIUM,
                'low': SeverityLevel.LOW
            }

            issue = IssueFinding(
                title=vuln.get('title', 'Unnamed issue'),
                description=vuln.get('description', str(vuln)),
                severity=severity_map.get(
                    vuln.get('severity', 'medium').lower(),
                    SeverityLevel.MEDIUM
                ),
                category=IssueCategory.LOGICAL_ERROR,
                confidence=red_team_result.confidence
            )
            issues.append(issue)

        # Generate fixes
        return self.blue_team.generate_solutions_with_ensemble(
            issues=issues,
            content=sop_content,
            content_type="technical_sop",
            api_key=self.api_key,
            model_name=self.blue_team_models[0],
            num_models=self.num_ensemble_models,
            strategy=BlueTeamStrategy.COMPREHENSIVE
        )

    def _run_evaluation(self, sop_content: str):
        """Run evaluator team assessment"""
        return self.evaluator_team.evaluate_with_ensemble(
            content=sop_content,
            content_type="technical_sop",
            api_key=self.api_key,
            model_name=self.evaluator_models[0],
            num_models=self.num_ensemble_models + 2  # More evaluators for rigor
        )


def main():
    """Command-line interface for SOP evolution"""

    parser = argparse.ArgumentParser(
        description="Evolve Standard Operating Procedures using ensemble methods"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input SOP file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output evolved SOP file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Maximum evolution iterations (default: 3)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Quality threshold to stop iteration (default: 0.90)"
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=7,
        help="Number of models in ensemble (default: 7)"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate versions after each iteration"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Define models to use
    red_team_models = ["gpt-4o", "claude-3-opus"]
    blue_team_models = ["gpt-4o", "claude-3-opus"]
    evaluator_models = ["gpt-4o", "claude-3-opus", "gemini-ultra"]

    # Create evolver
    evolver = SOPEvolver(
        api_key=api_key,
        red_team_models=red_team_models,
        blue_team_models=blue_team_models,
        evaluator_models=evaluator_models,
        num_ensemble_models=args.ensemble_size
    )

    # Run evolution
    try:
        results = evolver.evolve_sop(
            input_sop_path=args.input,
            output_sop_path=args.output,
            max_iterations=args.iterations,
            quality_threshold=args.threshold,
            save_intermediate=args.save_intermediate
        )

        print("\n✓ Evolution completed successfully!")
        sys.exit(0)

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"\n✗ Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

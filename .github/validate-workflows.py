#!/usr/bin/env python3
"""
RESE CI/CD Workflow Validator

This script validates all GitHub Actions workflow files for the RESE framework.
It checks YAML syntax, required structure, and best practices.

Usage:
    python .github/validate-workflows.py

Exit codes:
    0: All workflows are valid
    1: One or more workflows have errors
"""

import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any


class WorkflowValidator:
    """Validates GitHub Actions workflow files."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_workflow(self, workflow_path: Path) -> bool:
        """
        Validate a single workflow file.

        Args:
            workflow_path: Path to workflow YAML file

        Returns:
            True if workflow is valid
        """
        print(f"\nValidating: {workflow_path.name}")
        print("=" * 80)

        # Check file exists
        if not workflow_path.exists():
            self.errors.append(f"{workflow_path}: File not found")
            return False

        # Load YAML
        try:
            with open(workflow_path, encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"{workflow_path}: YAML error - {e}")
            return False
        except Exception as e:
            self.errors.append(f"{workflow_path}: Read error - {e}")
            return False

        # Validate structure
        valid = True

        # Check required fields
        if 'name' not in data:
            self.errors.append(f"{workflow_path}: Missing 'name' field")
            valid = False
        else:
            print(f"  Name: {data['name']}")

        if 'on' not in data:
            self.warnings.append(f"{workflow_path}: Missing 'on' field (triggers)")
        else:
            triggers = data['on']
            if isinstance(triggers, dict):
                trigger_count = len(triggers)
                print(f"  Triggers: {trigger_count} defined")
                for trigger in triggers.keys():
                    print(f"    - {trigger}")

        if 'jobs' not in data:
            self.errors.append(f"{workflow_path}: Missing 'jobs' field")
            return False

        jobs = data['jobs']
        job_count = len(jobs)
        print(f"  Jobs: {job_count} defined")

        # Validate each job
        for job_name, job_config in jobs.items():
            if not isinstance(job_config, dict):
                self.errors.append(f"{workflow_path}: Job '{job_name}' is not a dict")
                continue

            # Check required job fields
            if 'runs-on' not in job_config:
                self.warnings.append(f"{workflow_path}: Job '{job_name}' missing 'runs-on'")
            else:
                runner = job_config['runs-on']
                print(f"    - {job_name}: {runner}")

            # Check for timeout
            if 'timeout-minutes' not in job_config:
                self.warnings.append(f"{workflow_path}: Job '{job_name}' missing 'timeout-minutes'")

            # Check steps
            if 'steps' not in job_config:
                self.errors.append(f"{workflow_path}: Job '{job_name}' missing 'steps'")
            else:
                steps = job_config['steps']
                step_count = len(steps)
                print(f"      {step_count} steps")

        # Validate concurrency
        if 'concurrency' in data:
            concurrency = data['concurrency']
            if 'group' not in concurrency:
                self.warnings.append(f"{workflow_path}: Concurrency missing 'group'")
            if 'cancel-in-progress' not in concurrency:
                self.warnings.append(f"{workflow_path}: Concurrency missing 'cancel-in-progress'")

        # Validate env variables
        if 'env' in data:
            env_vars = data['env']
            print(f"  Environment Variables: {len(env_vars)} defined")

        print(f"  Status: {'VALID' if valid else 'INVALID'}")
        return valid

    def validate_all_workflows(self, workflows_dir: Path) -> bool:
        """
        Validate all workflow files in a directory.

        Args:
            workflows_dir: Path to .github/workflows directory

        Returns:
            True if all workflows are valid
        """
        print("RESE Framework CI/CD Workflow Validator")
        print("=" * 80)

        if not workflows_dir.exists():
            self.errors.append(f"Workflows directory not found: {workflows_dir}")
            return False

        # Find all workflow files
        workflow_files = sorted(workflows_dir.glob('*.yml')) + sorted(workflows_dir.glob('*.yaml'))

        if not workflow_files:
            self.errors.append(f"No workflow files found in {workflows_dir}")
            return False

        print(f"Found {len(workflow_files)} workflow file(s)")

        # Validate each workflow
        all_valid = True
        for workflow_file in workflow_files:
            if not self.validate_workflow(workflow_file):
                all_valid = False

        return all_valid

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\nAll workflows are valid! No errors or warnings found.")
        elif not self.errors:
            print(f"\nAll workflows are valid! {len(self.warnings)} warning(s) found.")
        else:
            print(f"\nValidation failed: {len(self.errors)} error(s), {len(self.warnings)} warning(s)")

        print()


def main():
    """Main entry point."""
    # Get repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    workflows_dir = repo_root / '.github' / 'workflows'

    # Validate workflows
    validator = WorkflowValidator()
    all_valid = validator.validate_all_workflows(workflows_dir)

    # Print summary
    validator.print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()

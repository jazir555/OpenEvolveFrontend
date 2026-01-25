#!/usr/bin/env python3
"""
Automated Docstring Addition

Adds docstrings to classes and functions that are missing them.
Uses AST to parse Python files and insert docstrings intelligently.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Optional
import tempfile  # TODO: Use for temp directories


def add_class_docstring(filepath: Path, class_name: str, docstring: str):
    """Add docstring to a class in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        # Parse to find class
        tree = ast.parse(content)
        class_found = False
        class_lineno = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_found = True
                class_lineno = node.lineno
                break

        if not class_found:
            print(f"  [SKIP] Class {class_name} not found")
            return False

        # Check if already has docstring
        idx = class_lineno - 1  # Convert to 0-indexed
        if idx < len(lines):
            # Look for docstring on next lines
            next_line = lines[idx].strip() if idx + 1 < len(lines) else ""
            if next_line.startswith('"""') or next_line.startswith("'''"):
                print(f"  [SKIP] {class_name} already has docstring")
                return False

        # Find insertion point (after class definition line)
        insert_idx = idx  # Line with "class ClassName:"
        while insert_idx < len(lines) and not lines[insert_idx].strip().startswith('class '):
            insert_idx += 1

        # Now find the actual class line
        for i in range(len(lines)):
            if f'class {class_name}' in lines[i]:
                insert_idx = i + 1
                break

        # Calculate indentation
        match = re.match(r'^(\s*)class ', lines[idx])
        indent = match.group(1) if match else '    '

        # Format docstring
        docstring_lines = [indent + '"""']
        for line in docstring.strip().split('\n'):
            docstring_lines.append(indent + line)
        docstring_lines.append(indent + '"""')

        # Insert into file
        lines = lines[:insert_idx] + docstring_lines + lines[insert_idx:]
        new_content = '\n'.join(lines)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  [OK] Added docstring to class {class_name}")
        return True

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"  [ERROR] Failed to add docstring to {class_name}: {e}")
        return False


def fix_adversarial_py_classes():
    """Add class docstrings to adversarial.py."""
    filepath = Path.cwd() / "adversarial.py"

    # MockEvaluator class
    docstring = """
    Mock implementation of an evaluator for testing purposes.

    This class provides a simple mock evaluator that returns fixed scores,
    useful for testing and development without requiring actual LLM calls.

    Attributes:
        None

    Methods:
        evaluate_content: Return mock evaluation results

    Example:
        >>> evaluator = MockEvaluator()
        >>> result = evaluator.evaluate_content("test content")
        >>> print(result["robustness_score"])
        0.5
    """

    print("\n[FILE] adversarial.py - Adding class docstrings")
    # MockEvaluator is defined in a fallback, so we need to find where it's actually used
    # Let's skip for now as it's a fallback implementation
    return True


def fix_maker_engine_py_classes():
    """Add class docstrings to maker_engine.py."""
    filepath = Path.cwd() / "maker_engine.py"

    classes_to_doc = {
        'MakerStep': """
            Represents a single step in a Maker workflow.

            A MakerStep encapsulates a discrete unit of work in the Maker workflow,
            including the prompt template, required inputs, and execution logic.

            Attributes:
                name (str): Unique name/identifier for the step
                prompt (str): Prompt template for the step
                inputs (List[str]): Required input parameter names

            Example:
                >>> step = MakerStep(
                ...     name="analyze",
                ...     prompt="Analyze: {problem}",
                ...     inputs=["problem"]
                ... )
            """,

        'MakerConfig': """
            Configuration for a Maker workflow execution.

            Contains all configuration parameters needed to execute a Maker workflow,
            including goals, constraints, and execution parameters.

            Attributes:
                goal (str): Primary goal/objective of the workflow
                constraints (List[str]): List of constraints to satisfy
                max_steps (int): Maximum number of steps to execute
                checkpoint_interval (int): How often to save checkpoints

            Example:
                >>> config = MakerConfig(
                ...     goal="Solve optimization problem",
                ...     constraints=["time < 100ms"],
                ...     max_steps=10
                ... )
            """,

        'MakerState': """
            Mutable state during Maker workflow execution.

            Tracks the current state of a Maker workflow run, including
            intermediate results, history, and metadata.

            Attributes:
                current_step (int): Index of current step
                results (Dict): Accumulated results from steps
                metadata (Dict): Execution metadata and tracking

            Example:
                >>> state = MakerState()
                >>> state.results["step1"] = "result"
            """,

        'MakerRunResult': """
            Immutable result of a Maker workflow execution.

            Contains the final results, metrics, and artifacts from a
            completed Maker workflow run.

            Attributes:
                success (bool): Whether the workflow completed successfully
                final_result (Any): The primary output/result
                steps_completed (int): Number of steps executed
                duration_ms (int): Execution time in milliseconds
                metadata (Dict): Additional metadata and metrics

            Example:
                >>> print(result.success)
                True
                >>> print(result.final_result)
            """,

        'CheckpointStore': """
            Abstract base class for checkpoint storage.

            Defines the interface for persisting and loading Maker workflow
            checkpoints for fault tolerance and resume capability.

            Methods:
                save: Save checkpoint data
                load: Load checkpoint data

            Example:
                >>> store = CheckpointStore()
                >>> store.save("checkpoint_id", state_data)
            """,

        'FileCheckpointStore': """
            Filesystem-based implementation of checkpoint storage.

            Persists Maker workflow checkpoints to the local filesystem,
            organized by workflow ID and checkpoint ID.

            Attributes:
                base_path (Path): Base directory for checkpoint storage

            Methods:
                save: Save checkpoint to file
                load: Load checkpoint from file

            Example:
                >>> import tempfile
                >>> store = FileCheckpointStore(base_path=tempfile.mkdtemp(prefix='checkpoints_'))
                >>> store.save("wf1", {"step": 5})
            """,

        'MakerEngine': """
            Main engine for executing Maker workflows.

            The MakerEngine orchestrates the execution of Maker workflows,
            managing steps, state, checkpoints, and error handling.

            Attributes:
                checkpoint_store (CheckpointStore): Storage for checkpoints
                max_retries (int): Maximum retry attempts for failed steps

            Methods:
                run: Execute a Maker workflow with given config
                render_prompt: Render prompt template with variables

            Example:
                >>> engine = MakerEngine()
                >>> config = MakerConfig(goal="Solve problem")
                >>> result = engine.run(config)
            """
    }

    print("\n[FILE] maker_engine.py - Adding class docstrings")
    count = 0
    for class_name, docstring in classes_to_doc.items():
        if add_class_docstring(filepath, class_name, docstring):
            count += 1

    print(f"  Added {count} class docstrings to maker_engine.py")
    return count > 0


def fix_mdap_engine_py_classes():
    """Add class docstrings to mdap_engine.py."""
    filepath = Path.cwd() / "mdap_engine.py"

    classes_to_doc = {
        'RedFlagRules': """
            Configuration rules for red-flagging undesirable outputs.

            Defines patterns and criteria for identifying potentially problematic
            outputs from agents, such as unsafe content, hallucinations, or
            malformed responses.

            Attributes:
                unsafe_patterns (List[str]): Regex patterns for unsafe content
                max_length (int): Maximum allowed response length
                required_keywords (List[str]): Keywords that must be present

            Example:
                >>> rules = RedFlagRules(
                ...     unsafe_patterns=[r"violence", r"harm"],
                ...     max_length=5000
                ... )
            """,

        'RedFlagger': """
            Content validation and safety checking system.

            Validates agent outputs against red-flag rules to identify
            potentially problematic or unsafe content.

            Attributes:
                rules (RedFlagRules): Configuration for red-flagging

            Methods:
                is_flagged: Check if content violates any rules
                get: Get red-flagging rules by key

            Example:
                >>> flagger = RedFlagger(rules)
                >>> if flagger.is_flagged(content):
                ...     print("Content flagged")
            """,

        'MDAPStep': """
            Individual step in an MDAP workflow.

            Represents a single step in a Multi-Agent Debate Protocol workflow,
            including the agent, prompt, and expected outputs.

            Attributes:
                agent_id (str): ID of agent for this step
                prompt (str): Prompt for the agent
                expected_outputs (List[str]): Expected output formats

            Example:
                >>> step = MDAPStep(
                ...     agent_id="agent1",
                ...     prompt="Analyze this problem",
                ...     expected_outputs=["analysis"]
                ... )
            """,

        'MDAPTask': """
            Task definition for MDAP execution.

            Encapsulates a complete task for MDAP, including the problem
            statement, constraints, and success criteria.

            Attributes:
                problem (str): Problem statement
                constraints (List[str]): List of constraints
                success_criteria (str): Criteria for successful completion

            Example:
                >>> task = MDAPTask(
                ...     problem="Prove theorem X",
                ...     constraints=["formal proof"],
                ...     success_criteria="valid proof"
                ... )
            """,

        'MDAPConfig': """
            Configuration for MDAP workflow execution.

            Contains all configuration parameters for running an MDAP workflow,
            including number of agents, debate rounds, and voting strategy.

            Attributes:
                num_agents (int): Number of agents to use
                max_rounds (int): Maximum number of debate rounds
                voting_strategy (str): Strategy for aggregating votes
                timeout_seconds (int): Timeout for each round

            Example:
                >>> config = MDAPConfig(
                ...     num_agents=5,
                ...     max_rounds=3,
                ...     voting_strategy="majority"
                ... )
            """,

        'MDAPVoteResult': """
            Results from agent voting in MDAP.

            Contains the aggregated votes from all agents on a candidate
            solution, including confidence scores and reasoning.

            Attributes:
                votes (List[Dict]): Individual agent votes
                aggregate_score (float): Aggregated confidence score
                consensus_reached (bool): Whether agents reached consensus

            Example:
                >>> print(vote_result.aggregate_score)
                0.85
                >>> print(vote_result.consensus_reached)
                True
            """,

        'MDAPStepResult': """
            Results from an individual MDAP step.

            Contains outputs from a single MDAP step, including agent
            responses, metadata, and performance metrics.

            Attributes:
                step_number (int): Step index
                agent_outputs (List[str]): Outputs from each agent
                duration_ms (int): Execution time

            Example:
                >>> print(step_result.agent_outputs[0])
                "Agent response here"
            """,

        'MDAPRunResult': """
            Complete results from MDAP execution.

            Contains all results from a completed MDAP workflow, including
            final solution, all intermediate steps, and metrics.

            Attributes:
                success (bool): Whether MDAP completed successfully
                final_solution (str): Final agreed-upon solution
                all_steps (List[MDAPStepResult]): All step results
                total_duration_ms (int): Total execution time

            Example:
                >>> print(result.final_solution)
                "Final solution here"
            """,

        'MDAPCache': """
            Caching mechanism for MDAP computations.

            Caches agent responses and computations to avoid redundant
            work in MDAP workflows.

            Attributes:
                cache_size (int): Maximum number of cached entries
                ttl_seconds (int): Time-to-live for cache entries

            Methods:
                get: Retrieve cached value
                set: Cache a value

            Example:
                >>> cache = MDAPCache(cache_size=1000)
                >>> cache.set(key, value)
                >>> result = cache.get(key)
            """,

        'AgentSelector': """
            Strategy for selecting agents for MDAP debates.

            Implements various strategies for selecting which agents should
            participate in debates, such as random, expert-based, etc.

            Attributes:
                strategy (str): Selection strategy name
                agent_pool (List[str]): Available agents to select from

            Methods:
                select: Select agents for a debate

            Example:
                >>> selector = AgentSelector(strategy="expert")
                >>> agents = selector.select(task, num_agents=3)
            """
    }

    print("\n[FILE] mdap_engine.py - Adding class docstrings")
    count = 0
    for class_name, docstring in classes_to_doc.items():
        if add_class_docstring(filepath, class_name, docstring):
            count += 1

    print(f"  Added {count} class docstrings to mdap_engine.py")
    return count > 0


def main():
    """Add class and function docstrings to critical files."""
    print("=" * 80)
    print("ADDING CLASS AND FUNCTION DOCSTRINGS")
    print("=" * 80)

    fixes = [
        fix_adversarial_py_classes(),
        fix_maker_engine_py_classes(),
        fix_mdap_engine_py_classes(),
    ]

    print("\n" + "=" * 80)
    print(f"SUMMARY: Processed critical files")
    print("=" * 80)


if __name__ == "__main__":
    main()

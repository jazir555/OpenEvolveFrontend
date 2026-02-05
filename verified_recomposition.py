"""
Verified Recomposition with LLM Judge + Algorithmic Verification

Architecture:
1. Ground Truth Store - Persist sub-solutions with content hashes
2. LLM Judge - Sees full content, outputs structured JSON decisions
3. Algorithmic Assembler - Executes JSON instructions verbatim
4. Verification Layer - Algorithmically checks all content preserved
5. Reject/Retry Loop - If verification fails, retry with feedback
"""

import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re

from ground_truth_store import GroundTruthStore, get_ground_truth_store

# ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_recomposition_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_recomposition_config = None

logger = logging.getLogger(__name__)


class AssemblyAction(Enum):
    """Types of assembly actions LLM can specify"""
    KEEP_VERBATIM = "keep_verbatim"  # Keep content exactly as-is
    MERGE = "merge"  # Merge with another sub-solution
    REORDER = "reorder"  # Change order within content
    EXTRACT = "extract"  # Extract specific parts
    SKIP = "skip"  # Skip this sub-solution


@dataclass
class AssemblyInstruction:
    """
    Single instruction for assembling one sub-solution.

    LLM outputs this in JSON format.
    """
    sub_problem_id: str
    action: AssemblyAction
    section_header: str  # Header to add before this section
    position: int  # Position in final assembly (0-indexed)
    merge_with: Optional[str] = None  # If action=MERGE, which sub-solution to merge with
    transformations: Optional[List[str]] = None  # List of transformations to apply
    transitions_before: Optional[str] = None  # Transition text to add before
    transitions_after: Optional[str] = None  # Transition text to add after
    preserve_integrity: bool = True  # Must preserve content integrity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['action'] = self.action.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssemblyInstruction':
        """Create from dictionary"""
        data = data.copy()
        data['action'] = AssemblyAction(data['action'])
        return cls(**data)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate instruction"""
        if self.action == AssemblyAction.MERGE and not self.merge_with:
            return False, "MERGE action requires merge_with field"

        if self.position < 0:
            return False, "Position must be >= 0"

        if not self.section_header:
            return False, "Section header is required"

        return True, None


@dataclass
class AssemblyPlan:
    """
    Complete assembly plan from LLM.

    Structured JSON output that guides algorithmic assembly.
    """
    instructions: List[AssemblyInstruction]
    intro: Optional[str] = None
    conclusion: Optional[str] = None
    global_transformations: Optional[List[str]] = None
    confidence_score: float = 0.0
    reasoning: Optional[str] = None  # LLM's reasoning for decisions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'instructions': [instr.to_dict() for instr in self.instructions],
            'intro': self.intro,
            'conclusion': self.conclusion,
            'global_transformations': self.global_transformations,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssemblyPlan':
        """Create from dictionary"""
        instructions = [
            AssemblyInstruction.from_dict(instr)
            for instr in data['instructions']
        ]

        return cls(
            instructions=instructions,
            intro=data.get('intro'),
            conclusion=data.get('conclusion'),
            global_transformations=data.get('global_transformations'),
            confidence_score=data.get('confidence_score', 0.0),
            reasoning=data.get('reasoning')
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate assembly plan"""
        errors = []

        if not self.instructions:
            errors.append("No instructions provided")

        positions = set()
        for instr in self.instructions:
            valid, error = instr.validate()
            if not valid:
                errors.append(f"{instr.sub_problem_id}: {error}")

            if instr.position in positions:
                errors.append(f"Duplicate position {instr.position}")
            positions.add(instr.position)

        # Check position continuity
        if positions:
            expected = set(range(len(positions)))
            if positions != expected:
                errors.append(f"Position gaps: expected {expected}, got {positions}")

        return len(errors) == 0, errors


class VerifiedRecomposer:
    """
    Recomposer with LLM Judge + Algorithmic Verification.

    Flow:
    1. Store sub-solutions in Ground Truth Store
    2. Send full content to LLM Judge
    3. LLM returns structured AssemblyPlan (JSON)
    4. Algorithmically assemble according to plan
    5. Verify all content preserved (algorithmic check)
    6. If verification fails -> retry with feedback
    """

    def __init__(
        self,
        ground_truth_store: Optional[GroundTruthStore] = None,
        max_retries: int = 3,
        strict_verification: bool = True
    ):
        """
        Initialize verified recomposer.

        Args:
            ground_truth_store: Store for ground truth (default: global instance)
            max_retries: Maximum retry attempts if verification fails
            strict_verification: If True, reject assembly if ANY content lost
        """
        self.ground_truth_store = ground_truth_store or get_ground_truth_store()
        self.max_retries = max_retries
        self.strict_verification = strict_verification

        # Initialize ROMA-MDAP-MAKER Engine for robust recomposition
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT recomposition preset for standardized high-reliability synthesis
                config_roma = get_recomposition_config(
                    mdap_max_samples=50,
                    mdap_min_confidence=0.4
                )
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config_roma)
                logger.info("ROMAMDAPMakerAssociativeEngine initialized for VerifiedRecomposer")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to initialize ROMA engine: {e}")

    def store_ground_truth(
        self,
        sub_problem_id: str,
        description: str,
        dependencies: List[str],
        solution_content: str,
        metadata: Dict[str, Any],
        source: str = "llm"
    ) -> str:
        """
        Store sub-solution as ground truth.

        Args:
            sub_problem_id: Unique identifier
            description: Sub-problem description
            dependencies: List of dependencies
            solution_content: Solution content
            metadata: Additional metadata
            source: Source of solution

        Returns:
            Content hash
        """
        ground_truth = self.ground_truth_store.store_sub_solution(
            sub_problem_id=sub_problem_id,
            description=description,
            dependencies=dependencies,
            solution_content=solution_content,
            metadata=metadata,
            source=source
        )

        logger.info(f"Stored ground truth: {sub_problem_id} (hash: {ground_truth.content_hash[:16]}...)")

        return ground_truth.content_hash

    def create_llm_judge_prompt(
        self,
        sub_solutions: Dict[str, Any],
        conflicts: List[Any],
        context: str
    ) -> str:
        """
        Create prompt for LLM Judge.

        LLM sees FULL content to make accurate decisions.

        Args:
            sub_solutions: Dict of sub-solutions (full content)
            conflicts: List of conflicts
            context: Additional context

        Returns:
            Prompt for LLM
        """
        # Build sub-solution summaries WITH FULL CONTENT
        solutions_text = ""
        for sub_id, solution in sub_solutions.items():
            content = solution.get('solution_content', '')
            solutions_text += f"""
[{sub_id}]
Description: {solution.get('description', 'N/A')}
Confidence: {solution.get('confidence_score', 0.0):.2f}
Dependencies: {', '.join(solution.get('dependencies', []))}

FULL CONTENT:
```
{content}
```

"""

        # Build conflict summary
        conflicts_text = ""
        if conflicts:
            conflicts_text = "\nCONFLICTS TO ADDRESS:\n"
            for conflict in conflicts:
                conflicts_text += f"- {conflict.get('conflict_type')}: {conflict.get('description')}\n"

        # Build the prompt
        prompt = f"""You are an expert solution integrator. Your task is to analyze sub-solutions and create a precise assembly plan.

{context}

{solutions_text}
{conflicts_text}

YOUR TASK - CREATE STRUCTURED ASSEMBLY PLAN:

Analyze the sub-solutions above and output a JSON assembly plan with this exact structure:

{{
    "instructions": [
        {{
            "sub_problem_id": "sol_1",
            "action": "keep_verbatim",
            "section_header": "Authentication System",
            "position": 0,
            "preserve_integrity": true,
            "transitions_before": null,
            "transitions_after": "Now that authentication is established..."
        }},
        {{
            "sub_problem_id": "sol_2",
            "action": "keep_verbatim",
            "section_header": "User Profile Management",
            "position": 1,
            "preserve_integrity": true,
            "transitions_before": "Building on the authentication layer...",
            "transitions_after": null
        }}
    ],
    "intro": "This document presents a complete user management system...",
    "conclusion": "All components work together to provide secure user management.",
    "global_transformations": null,
    "confidence_score": 0.95,
    "reasoning": "Placing authentication first as it's the foundation. Profile management follows naturally."
}}

ACTION TYPES:
- keep_verbatim: Keep content exactly as-is (RECOMMENDED for code, APIs)
- merge: Merge content with another sub-solution (USE WITH CAUTION)
- reorder: Change internal order of content
- extract: Extract only specific parts
- skip: Skip this sub-solution entirely

CRITICAL RULES:
1. Default to "keep_verbatim" with preserve_integrity=true
2. Use "merge" ONLY when content truly duplicates (e.g., same function defined twice)
3. NEVER drop critical content (code, API definitions, requirements)
4. Position must be sequential starting from 0
5. All sub-problems must have an instruction
6. Transitions should be brief (1-2 sentences)

OUTPUT ONLY THE JSON. No additional text."""

        return prompt

    def parse_llm_response(self, llm_response: str) -> Tuple[Optional[AssemblyPlan], List[str]]:
        """
        Parse LLM response into AssemblyPlan.

        Args:
            llm_response: Raw LLM response

        Returns:
            Tuple of (AssemblyPlan or None, list of parse errors)
        """
        errors = []

        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    errors.append("No JSON found in LLM response")
                    return None, errors

            # Parse JSON
            data = json.loads(json_str)

            # Create AssemblyPlan
            plan = AssemblyPlan.from_dict(data)

            # Validate plan
            valid, validation_errors = plan.validate()
            if not valid:
                errors.extend(validation_errors)
                return None, errors

            logger.info(f"Parsed valid assembly plan with {len(plan.instructions)} instructions")
            return plan, []

        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")
            return None, errors
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            errors.append(f"Parse error: {e}")
            return None, errors

    def assemble_from_plan(
        self,
        plan: AssemblyPlan,
        sub_solutions: Dict[str, Any]
    ) -> str:
        """
        Algorithmically assemble solution according to plan.

        This is the DETERMINISTIC assembly step.

        Args:
            plan: Assembly plan from LLM
            sub_solutions: Dict of sub-solutions

        Returns:
            Assembled content
        """
        parts = []

        # Add intro if provided
        if plan.intro:
            parts.append(plan.intro)
            parts.append("")

        # Sort instructions by position
        sorted_instructions = sorted(plan.instructions, key=lambda x: x.position)

        # Execute each instruction
        for instr in sorted_instructions:
            sub_id = instr.sub_problem_id

            if sub_id not in sub_solutions:
                logger.warning(f"Sub-problem {sub_id} not found, skipping")
                continue

            solution = sub_solutions[sub_id]
            content = solution.get('solution_content', '')

            # Add transition before
            if instr.transitions_before:
                parts.append(instr.transitions_before)
                parts.append("")

            # Add section header
            parts.append(f"## {instr.section_header}")
            parts.append("")

            # Execute action
            if instr.action == AssemblyAction.KEEP_VERBATIM:
                # Insert content VERBATIM
                parts.append(content)
                logger.info(f"Inserted {sub_id} verbatim at position {instr.position}")

            elif instr.action == AssemblyAction.SKIP:
                logger.info(f"Skipped {sub_id} per instruction")
                continue

            elif instr.action == AssemblyAction.MERGE:
                # For now, just keep verbatim but log warning
                logger.warning(f"MERGE action for {sub_id} - keeping verbatim (merge not yet implemented)")
                parts.append(content)

            elif instr.action == AssemblyAction.EXTRACT:
                logger.warning(f"EXTRACT action for {sub_id} - keeping verbatim (extract not yet implemented)")
                parts.append(content)

            else:
                logger.warning(f"Unknown action {instr.action} for {sub_id}, keeping verbatim")
                parts.append(content)

            # Add transition after
            if instr.transitions_after:
                parts.append("")
                parts.append(instr.transitions_after)

            parts.append("")

        # Add conclusion if provided
        if plan.conclusion:
            parts.append(plan.conclusion)
            parts.append("")

        assembled = '\n'.join(parts)

        logger.info(f"Assembled {len(assembled)} chars from {len(sorted_instructions)} instructions")
        return assembled

    def verify_assembly(
        self,
        assembled_content: str,
        sub_problem_ids: List[str]
    ) -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
        """
        Verify that all content is preserved in assembled output.

        Algorithmic verification against ground truth.

        Args:
            assembled_content: Final assembled content
            sub_problem_ids: List of sub-problem IDs to verify

        Returns:
            Tuple of (all_preserved, verification_results)
        """
        verification_results = self.ground_truth_store.verify_all_solutions_preserved(
            assembled_output=assembled_content,
            sub_problem_ids=sub_problem_ids
        )

        all_preserved = all(preserved for preserved, _ in verification_results.values())

        return all_preserved, verification_results

    def recompose_with_verification(
        self,
        sub_solutions: Dict[str, Any],
        conflicts: List[Any],
        context: str,
        llm_call_fn: callable  # Function to call LLM
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Recompose with automatic verification and retry loop.

        Args:
            sub_solutions: Dict of sub-solutions
            conflicts: List of conflicts
            context: Additional context
            llm_call_fn: Function to call LLM (takes prompt, returns response)

        Returns:
            Tuple of (assembled_content or None, metadata)
        """
        # Step 1: Store all as ground truth
        sub_problem_ids = []
        for sub_id, solution in sub_solutions.items():
            self.store_ground_truth(
                sub_problem_id=sub_id,
                description=solution.get('description', ''),
                dependencies=solution.get('dependencies', []),
                solution_content=solution.get('solution_content', ''),
                metadata=solution,
                source='llm'
            )
            sub_problem_ids.append(sub_id)

        # Retry loop
        for attempt in range(self.max_retries):
            logger.info(f"Recomposition attempt {attempt + 1}/{self.max_retries}")

            # Step 2: Get assembly plan from LLM
            prompt = self.create_llm_judge_prompt(sub_solutions, conflicts, context)

            try:
                llm_response = llm_call_fn(prompt)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"LLM call failed: {e}")
                continue

            # Step 3: Parse LLM response
            plan, parse_errors = self.parse_llm_response(llm_response)

            if plan is None:
                logger.error(f"Failed to parse LLM response: {parse_errors}")
                if attempt < self.max_retries - 1:
                    # Provide feedback in next attempt
                    context += f"\n\nFEEDBACK: Previous attempt failed. Errors: {parse_errors}"
                continue

            # Step 4: Assemble from plan
            assembled = self.assemble_from_plan(plan, sub_solutions)

            # Step 5: Verify assembly
            all_preserved, verification_results = self.verify_assembly(
                assembled,
                sub_problem_ids
            )

            metadata = {
                'attempt': attempt + 1,
                'llm_plan_confidence': plan.confidence_score,
                'llm_reasoning': plan.reasoning,
                'verification_results': verification_results,
                'all_preserved': all_preserved,
                'assembly_plan': plan.to_dict()
            }

            if all_preserved:
                logger.info("[OK] Verification PASSED - all content preserved")
                return assembled, metadata
            else:
                logger.error(f"[FAIL] Verification FAILED - attempt {attempt + 1}")

                # Add feedback for next attempt
                failed_ids = [
                    sub_id for sub_id, (preserved, _) in verification_results.items()
                    if not preserved
                ]

                if attempt < self.max_retries - 1:
                    context += f"\n\nFEEDBACK: Verification failed. Content not preserved for: {failed_ids}"
                    context += "\n\nCRITICAL: You MUST use 'keep_verbatim' action for these sub-problems."
                    context += "\nDo NOT skip or merge these solutions."

        # All retries exhausted
        logger.error(f"Failed after {self.max_retries} attempts")
        return None, {
            'error': 'max_retries_exceeded',
            'final_verification_results': verification_results
        }


# Convenience function
def recompose_with_verification(
    sub_solutions: Dict[str, Any],
    conflicts: List[Any],
    context: str,
    llm_call_fn: callable,
    ground_truth_store: Optional[GroundTruthStore] = None
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Convenience function for verified recomposition.

    Args:
        sub_solutions: Dict of sub-solutions
        conflicts: List of conflicts
        context: Additional context
        llm_call_fn: Function to call LLM
        ground_truth_store: Optional ground truth store

    Returns:
        Tuple of (assembled_content or None, metadata)
    """
    recomposer = VerifiedRecomposer(ground_truth_store=ground_truth_store)
    return recomposer.recompose_with_verification(
        sub_solutions=sub_solutions,
        conflicts=conflicts,
        context=context,
        llm_call_fn=llm_call_fn
    )

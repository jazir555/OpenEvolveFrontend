"""
Associative Recomposition System with AgentJSON Integration

Domain-agnostic system where:
1. LLM classifies problem domain and type (no hardcoded triggers)
2. LLM outputs structured JSON via AgentJSON (robust parsing)
3. Algorithmic verification ensures content preserved
4. LLM acts as final judge of correctness (domain-specific)

Architecture:
- Generative Layer: LLM reasoning and classification
- Predictive Layer: Structured JSON output with AgentJSON
- Algorithmic Layer: Content preservation verification
- Judgment Layer: LLM correctness evaluation
"""

import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

try:
    from agentjson.src.json_prob_parser import parse as agentjson_parse
    from agentjson.src.json_prob_parser.types import RepairOptions, RepairResult
    AGENTJSON_AVAILABLE = True
except ImportError:
    AGENTJSON_AVAILABLE = False
    logging.warning("AgentJSON not available, falling back to json.loads")

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


class SolutionType(Enum):
    """Common solution output types"""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    API_SPEC = "api_spec"
    DATA_MODEL = "data_model"
    ARCHITECTURE = "architecture"
    WORKFLOW = "workflow"
    ANALYSIS = "analysis"
    REPORT = "report"
    TUTORIAL = "tutorial"
    OTHER = "other"


class ProblemDomain(Enum):
    """Problem domains (extensible, not hardcoded)"""
    SOFTWARE_DEVELOPMENT = "software_development"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    DEVOPS = "devops"
    SECURITY = "security"
    BUSINESS = "business"
    RESEARCH = "research"
    EDUCATION = "education"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    OTHER = "other"


@dataclass
class DomainClassification:
    """
    LLM-provided domain classification.

    LLM determines what type of problem this is (domain-agnostic).
    """
    problem_type: str  # Free-form classification by LLM
    domain: ProblemDomain  # Categorized domain
    solution_type: SolutionType  # Type of output expected
    field: str  # Specific field (e.g., "web development", "computer vision")
    complexity: str  # "low", "medium", "high", "expert"
    confidence: float  # LLM's confidence in classification
    reasoning: str  # Why LLM classified this way

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['domain'] = self.domain.value
        data['solution_type'] = self.solution_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainClassification':
        """Create from dictionary"""
        data = data.copy()
        data['domain'] = ProblemDomain(data.get('domain', 'other'))
        data['solution_type'] = SolutionType(data.get('solution_type', 'other'))
        return cls(**data)


@dataclass
class AssemblyInstruction:
    """
    Assembly instruction for a sub-solution.
    """
    sub_problem_id: str
    sub_problem_identity: str  # LLM-provided identity (what this component IS)
    action: str  # "keep_verbatim", "merge", "reorder", "skip"
    section_header: str
    position: int
    preserve_integrity: bool = True  # Must preserve content exactly
    merge_with: Optional[str] = None
    transformations: Optional[List[str]] = None
    transition_before: Optional[str] = None
    transition_after: Optional[str] = None
    notes: Optional[str] = None  # LLM notes about this component

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssemblyInstruction':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class AssemblyPlanJSON:
    """
    Complete assembly plan in structured JSON.

    LLM outputs this structure - domain-agnostic and flexible.
    """
    # Classification (LLM determines)
    classification: DomainClassification

    # Target solution specification
    target_solution_type: SolutionType
    target_solution_description: str  # What the final output should be
    success_criteria: List[str]  # How to judge success

    # Sub-problem identities (LLM-provided)
    sub_problem_identities: Dict[str, str]  # ID → what this component is

    # Assembly instructions
    instructions: List[AssemblyInstruction]

    # Structure
    intro: Optional[str] = None
    conclusion: Optional[str] = None
    global_notes: Optional[str] = None

    # Metadata
    confidence_score: float = 0.0
    reasoning: str = ""  # LLM's reasoning for assembly decisions
    estimated_quality: str = ""  # "low", "medium", "high", "excellent"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'classification': self.classification.to_dict(),
            'target_solution_type': self.target_solution_type.value,
            'target_solution_description': self.target_solution_description,
            'success_criteria': self.success_criteria,
            'sub_problem_identities': self.sub_problem_identities,
            'instructions': [instr.to_dict() for instr in self.instructions],
            'intro': self.intro,
            'conclusion': self.conclusion,
            'global_notes': self.global_notes,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'estimated_quality': self.estimated_quality
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssemblyPlanJSON':
        """Create from dictionary"""
        data = data.copy()
        data['classification'] = DomainClassification.from_dict(data['classification'])
        data['target_solution_type'] = SolutionType(data['target_solution_type'])
        data['instructions'] = [
            AssemblyInstruction.from_dict(instr)
            for instr in data['instructions']
        ]
        return cls(**data)


class AssociativeRecomposer:
    """
    Domain-associative recomposer using AgentJSON.

    Blends generative LLM reasoning with algorithmic verification.
    """

    def __init__(
        self,
        ground_truth_store: Optional[GroundTruthStore] = None,
        use_agentjson: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize associative recomposer.

        Args:
            ground_truth_store: Store for ground truth
            use_agentjson: Use AgentJSON for robust JSON parsing
            max_retries: Maximum retry attempts
        """
        self.ground_truth_store = ground_truth_store or get_ground_truth_store()
        self.use_agentjson = use_agentjson and AGENTJSON_AVAILABLE
        self.max_retries = max_retries

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
                logger.info("ROMAMDAPMakerAssociativeEngine initialized for AssociativeRecomposer")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to initialize ROMA engine: {e}")

        if self.use_agentjson:
            logger.info("Using AgentJSON for robust JSON parsing")
        else:
            logger.info("Using standard JSON parsing")

    def create_associative_prompt(
        self,
        sub_solutions: Dict[str, Any],
        conflicts: List[Any],
        problem_statement: str
    ) -> str:
        """
        Create domain-agnostic associative prompt.

        LLM classifies the problem itself - no hardcoded triggers.

        Args:
            sub_solutions: Dict of sub-solutions
            conflicts: List of conflicts
            problem_statement: Original problem statement

        Returns:
            Prompt for LLM
        """
        # Build sub-solution summary with FULL content
        solutions_text = ""
        for sub_id, solution in sub_solutions.items():
            content = solution.get('solution_content', '')
            solutions_text += f"""
[{sub_id}]
Confidence: {solution.get('confidence_score', 0.0):.2f}

FULL CONTENT:
{content}

---

"""

        # Build the prompt
        prompt = f"""You are an expert solution integrator and analyst. Your task is to:
1. CLASSIFY the problem domain and type
2. ANALYZE each sub-solution's identity and purpose
3. CREATE a precise assembly plan in JSON

PROBLEM STATEMENT:
{problem_statement}

{solutions_text}

CONFLICTS:
{chr(10).join(f'- {c.get("description", "Unknown conflict")}' for c in conflicts)}

YOUR TASK - OUTPUT JSON:

Analyze the problem and sub-solutions, then output this exact JSON structure:

{{
    "classification": {{
        "problem_type": "What specific type of problem is this? (free-form)",
        "domain": "software_development|data_science|machine_learning|devops|security|business|research|education|legal|healthcare|finance|other",
        "solution_type": "code|documentation|configuration|api_spec|data_model|architecture|workflow|analysis|report|tutorial|other",
        "field": "What specific field? (e.g., 'web authentication', 'computer vision', 'data pipeline')",
        "complexity": "low|medium|high|expert",
        "confidence": 0.95,
        "reasoning": "Explain why you classified this way"
    }},
    "target_solution_type": "code|documentation|configuration|api_spec|data_model|architecture|workflow|analysis|report|tutorial|other",
    "target_solution_description": "Describe what the final assembled solution should be",
    "success_criteria": [
        "Criterion 1 for judging success",
        "Criterion 2 for judging success",
        "Criterion 3 for judging success"
    ],
    "sub_problem_identities": {{
        "sol_1": "What is this component? (e.g., 'JWT authentication module')",
        "sol_2": "What is this component? (e.g., 'User profile data model')",
        "sol_3": "What is this component? (e.g., 'Role-based access control')"
    }},
    "instructions": [
        {{
            "sub_problem_id": "sol_1",
            "sub_problem_identity": "Brief reminder of what this is",
            "action": "keep_verbatim|merge|reorder|skip",
            "section_header": "Header for this section",
            "position": 0,
            "preserve_integrity": true,
            "merge_with": null,
            "transformations": null,
            "transition_before": null,
            "transition_after": "Transition to next section",
            "notes": "Any notes about this component"
        }}
    ],
    "intro": "Brief introduction (2-3 sentences)",
    "conclusion": "Brief conclusion (2-3 sentences)",
    "global_notes": "Any important notes about the assembly",
    "confidence_score": 0.95,
    "reasoning": "Explain your assembly strategy and why it's optimal",
    "estimated_quality": "low|medium|high|excellent"
}}

CRITICAL RULES:
1. Classify HONESTLY - use your judgment to identify domain and type
2. Default action is "keep_verbatim" with preserve_integrity=true
3. Only use "merge" if content genuinely duplicates
4. NEVER use "skip" unless sub-solution is truly irrelevant
5. Position must be sequential from 0
6. Provide honest confidence scores

OUTPUT ONLY THE JSON. No markdown, no explanation."""

        return prompt

    def parse_llm_response(
        self,
        llm_response: str
    ) -> Tuple[Optional[AssemblyPlanJSON], List[str]]:
        """
        Parse LLM response using AgentJSON or fallback.

        Args:
            llm_response: Raw LLM response

        Returns:
            Tuple of (AssemblyPlanJSON or None, list of errors)
        """
        errors = []

        if self.use_agentjson:
            return self._parse_with_agentjson(llm_response)
        else:
            return self._parse_with_json(llm_response)

    def _parse_with_agentjson(
        self,
        llm_response: str
    ) -> Tuple[Optional[AssemblyPlanJSON], List[str]]:
        """
        Parse using AgentJSON with probabilistic repair.

        Handles malformed JSON, incomplete output, etc.
        """
        errors = []

        try:
            # Configure AgentJSON for robust parsing
            options = RepairOptions(
                mode="probabilistic",
                top_k=3,  # Get top 3 candidates
                beam_width=32,
                max_repairs=50,
                allow_comments=True,
                allow_single_quotes=True,
                allow_unquoted_keys=True,
                allow_trailing_commas=True,
                partial_ok=True  # Accept partial results
            )

            # Parse with AgentJSON
            result: RepairResult = agentjson_parse(
                llm_response,
                options=options
            )

            if result.status == "failed" or result.best is None:
                errors.append(f"AgentJSON parsing failed: {result.errors}")
                # Try extracting JSON manually as fallback
                return self._extract_json_manually(llm_response)

            # Get best candidate
            candidate = result.best

            if candidate.value is None:
                errors.append("AgentJSON produced None value")
                return None, errors

            # Validate structure
            if not isinstance(candidate.value, dict):
                errors.append(f"Expected dict, got {type(candidate.value)}")
                return None, errors

            # Create AssemblyPlanJSON
            plan = AssemblyPlanJSON.from_dict(candidate.value)

            logger.info(f"✓ AgentJSON parsing successful (confidence: {candidate.confidence:.2f})")
            logger.info(f"  Repairs: {len(candidate.repairs)}")

            return plan, []

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            errors.append(f"AgentJSON error: {e}")
            logger.warning(f"AgentJSON failed, trying manual extraction: {e}")
            return self._extract_json_manually(llm_response)

    def _parse_with_json(
        self,
        llm_response: str
    ) -> Tuple[Optional[AssemblyPlanJSON], List[str]]:
        """
        Parse using standard JSON (fallback).
        """
        errors = []
        import re

        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                errors.append("No JSON found in response")
                return None, errors

            json_str = json_match.group(0)
            data = json.loads(json_str)

            plan = AssemblyPlanJSON.from_dict(data)
            return plan, []

        except json.JSONDecodeError as e:
            errors.append(f"JSON decode error: {e}")
            return None, errors
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            errors.append(f"Parse error: {e}")
            return None, errors

    def _extract_json_manually(
        self,
        llm_response: str
    ) -> Tuple[Optional[AssemblyPlanJSON], List[str]]:
        """
        Manual JSON extraction as last resort.
        """
        errors = []
        import re

        try:
            # Try to find JSON object
            match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', llm_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                plan = AssemblyPlanJSON.from_dict(data)
                return plan, []

            errors.append("Could not extract JSON from response")
            return None, errors

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            errors.append(f"Manual extraction failed: {e}")
            return None, errors

    def assemble_from_plan(
        self,
        plan: AssemblyPlanJSON,
        sub_solutions: Dict[str, Any]
    ) -> str:
        """
        Algorithmically assemble from plan.

        Args:
            plan: Assembly plan from LLM
            sub_solutions: Dict of sub-solutions

        Returns:
            Assembled content
        """
        parts = []

        # Add intro
        if plan.intro:
            parts.append(plan.intro)
            parts.append("")

        # Sort by position
        sorted_instructions = sorted(plan.instructions, key=lambda x: x.position)

        for instr in sorted_instructions:
            sub_id = instr.sub_problem_id

            if sub_id not in sub_solutions:
                logger.warning(f"Sub-problem {sub_id} not found")
                continue

            solution = sub_solutions[sub_id]
            content = solution.get('solution_content', '')

            # Transition before
            if instr.transition_before:
                parts.append(instr.transition_before)
                parts.append("")

            # Header
            parts.append(f"## {instr.section_header}")
            parts.append("")

            # Action
            if instr.action == "keep_verbatim":
                parts.append(content)
                logger.info(f"✓ {sub_id} ({instr.sub_problem_identity}): kept verbatim")

            elif instr.action == "skip":
                logger.info(f"⊘ {sub_id} ({instr.sub_problem_identity}): skipped")
                continue

            elif instr.action == "merge":
                logger.warning(f"⚠ {sub_id}: merge not implemented, keeping verbatim")
                parts.append(content)

            else:
                logger.warning(f"⚠ {sub_id}: unknown action {instr.action}, keeping verbatim")
                parts.append(content)

            # Transition after
            if instr.transition_after:
                parts.append("")
                parts.append(instr.transition_after)

            parts.append("")

        # Conclusion
        if plan.conclusion:
            parts.append(plan.conclusion)

        assembled = '\n'.join(parts)

        logger.info(f"Assembled {len(assembled)} chars from {len(sorted_instructions)} components")
        return assembled

    def llm_judgment_prompt(
        self,
        assembled_content: str,
        plan: AssemblyPlanJSON,
        sub_solutions: Dict[str, Any]
    ) -> str:
        """
        Create prompt for LLM judgment of correctness.

        LLM judges if the reassembled solution is correct and complete.

        Args:
            assembled_content: Final assembled solution
            plan: Assembly plan used
            sub_solutions: Original sub-solutions

        Returns:
            Judgment prompt for LLM
        """
        prompt = f"""You are an expert solution reviewer. Your task is to JUDGE if the reassembled solution is correct.

CLASSIFICATION:
- Domain: {plan.classification.domain.value}
- Type: {plan.classification.solution_type.value}
- Field: {plan.classification.field}
- Complexity: {plan.classification.complexity}

TARGET SOLUTION:
{plan.target_solution_description}

SUCCESS CRITERIA:
{chr(10).join(f'- {c}' for c in plan.success_criteria)}

REASSEMBLED SOLUTION:
{assembled_content}

SUB-SOLUTIONS PROVIDED:
{len(sub_solutions)} sub-solutions were provided for assembly

ASSEMBLY STRATEGY:
{plan.reasoning}

JUDGMENT TASK:
Evaluate the reassembled solution against the success criteria.

Output JSON:
{{
    "is_correct": true|false,
    "completeness_score": 0.95,
    "quality_score": 0.90,
    "missing_elements": ["element1", "element2"],
    "issues": ["Issue 1", "Issue 2"],
    "strengths": ["Strength 1", "Strength 2"],
    "verdict": "excellent|good|acceptable|needs_improvement|unacceptable",
    "confidence": 0.95,
    "reasoning": "Detailed explanation of your judgment"
}}

Be HONEST and THOROUGH. If content is missing or incorrect, say so.

OUTPUT ONLY THE JSON."""

        return prompt

    def parse_judgment(
        self,
        llm_response: str
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Parse LLM judgment response.

        Args:
            llm_response: LLM judgment response

        Returns:
            Tuple of (judgment dict or None, errors)
        """
        if self.use_agentjson:
            # Use AgentJSON for robust parsing
            return self._parse_judgment_with_agentjson(llm_response)
        else:
            return self._parse_judgment_with_json(llm_response)

    def _parse_judgment_with_agentjson(
        self,
        llm_response: str
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Parse judgment using AgentJSON"""
        errors = []

        try:
            options = RepairOptions(
                mode="probabilistic",
                top_k=1,
                partial_ok=True
            )

            result = agentjson_parse(llm_response, options=options)

            if result.status == "failed" or result.best is None:
                return None, ["Failed to parse judgment"]

            judgment = result.best.value
            logger.info(f"✓ Judgment parsed: is_correct={judgment.get('is_correct', False)}")
            return judgment, []

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            errors.append(f"Judgment parse error: {e}")
            return None, errors

    def _parse_judgment_with_json(
        self,
        llm_response: str
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Parse judgment using standard JSON"""
        errors = []
        import re

        try:
            match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not match:
                return None, ["No JSON in judgment"]

            judgment = json.loads(match.group(0))
            return judgment, []

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            errors.append(f"Judgment JSON error: {e}")
            return None, errors

    def recompose_with_verification(
        self,
        sub_solutions: Dict[str, Any],
        conflicts: List[Any],
        problem_statement: str,
        llm_call_fn: Callable[[str], str]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Full recomposition pipeline with verification.

        Args:
            sub_solutions: Dict of sub-solutions
            conflicts: List of conflicts
            problem_statement: Original problem
            llm_call_fn: Function to call LLM (prompt → response)

        Returns:
            Tuple of (assembled_content or None, metadata)
        """
        metadata = {
            'attempts': [],
            'classification': None,
            'judgment': None,
            'verification_results': None
        }

        # Store ground truth
        sub_problem_ids = []
        for sub_id, solution in sub_solutions.items():
            self.ground_truth_store.store_sub_solution(
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
            logger.info(f"\n{'='*70}")
            logger.info(f"Recomposition Attempt {attempt + 1}/{self.max_retries}")
            logger.info(f"{'='*70}")

            attempt_metadata = {'attempt': attempt + 1}

            # Step 1: Get assembly plan from LLM
            prompt = self.create_associative_prompt(
                sub_solutions, conflicts, problem_statement
            )

            try:
                llm_response = llm_call_fn(prompt)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"LLM call failed: {e}")
                attempt_metadata['error'] = str(e)
                metadata['attempts'].append(attempt_metadata)
                continue

            # Step 2: Parse plan
            plan, parse_errors = self.parse_llm_response(llm_response)

            if plan is None:
                logger.error(f"Failed to parse plan: {parse_errors}")
                attempt_metadata['parse_errors'] = parse_errors
                metadata['attempts'].append(attempt_metadata)
                continue

            # Store classification
            metadata['classification'] = plan.classification.to_dict()
            logger.info(f"Domain: {plan.classification.domain.value}")
            logger.info(f"Type: {plan.classification.solution_type.value}")
            logger.info(f"Field: {plan.classification.field}")

            # Step 3: Assemble
            assembled = self.assemble_from_plan(plan, sub_solutions)

            # Step 4: Algorithmic verification (content preserved)
            logger.info("\n--- Algorithmic Verification ---")
            verification_results = self.ground_truth_store.verify_all_solutions_preserved(
                assembled, sub_problem_ids
            )

            # Check if all solutions are preserved
            all_preserved = all(preserved for preserved, _ in verification_results.values())

            metadata['verification_results'] = verification_results

            if not all_preserved:
                logger.error("✗ Algorithmic verification FAILED - content missing")
                # Don't proceed to LLM judgment if algorithmic check fails
                continue

            logger.info("✓ Algorithmic verification PASSED - all content preserved")

            # Step 5: LLM judgment (correctness)
            logger.info("\n--- LLM Judgment ---")
            judgment_prompt = self.llm_judgment_prompt(assembled, plan, sub_solutions)

            try:
                judgment_response = llm_call_fn(judgment_prompt)
                judgment, judgment_errors = self.parse_judgment(judgment_response)

                if judgment:
                    metadata['judgment'] = judgment
                    logger.info(f"Is Correct: {judgment.get('is_correct', False)}")
                    logger.info(f"Verdict: {judgment.get('verdict', 'unknown')}")
                    logger.info(f"Quality: {judgment.get('quality_score', 0):.2f}")

                    if judgment.get('is_correct', False):
                        logger.info("✓ LLM judgment PASSED - solution is correct")
                        # Store the successful plan in metadata for downstream use
                        metadata['plan'] = plan.to_dict()
                        return assembled, metadata
                    else:
                        logger.warning("✗ LLM judgment FAILED")
                        logger.warning(f"Reasoning: {judgment.get('reasoning', 'N/A')}")
                else:
                    logger.warning(f"Could not parse judgment: {judgment_errors}")

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"LLM judgment failed: {e}")

            attempt_metadata['plan'] = plan.to_dict()
            metadata['attempts'].append(attempt_metadata)

        logger.error(f"\n{'='*70}")
        logger.error(f"Failed after {self.max_retries} attempts")
        logger.error(f"{'='*70}")

        return None, metadata

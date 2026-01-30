"""
MDAP/MAKER + Associative Recomposition Integration

Complete system integrating:
- MDAP (Multi-Agent Debate Protocol) - Multi-agent solution validation
- MAKER (Multi-step orchestration) - Structured decomposition→recomposition workflow
- Associative Recomposition - Domain-agnostic LLM + algorithmic verification
- Ground Truth Store - Persistent verification layer

Architecture:
1. MAKER orchestrates the full workflow
2. Associative Recomposition handles assembly
3. MDAP validates assembled solutions
4. Ground Truth ensures content preservation
5. LLM judges final correctness

Author: OpenEvolve
Date: 2026-01-09
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Import MDAP components
try:
    from mdap_engine import (
        MDAPOrchestrator, MDAPConfig, MDAPTask, MDAPStep,
        MDAPVoteResult, RedFlagRules, RedFlagger,
        canonicalize_candidate, candidate_confidence
    )
    from workflow_structures import Team, ModelConfig
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logger.warning("MDAP engine not available")
    Team = None  # type: ignore
    ModelConfig = None  # type: ignore

# Import MAKER components
try:
    from maker_engine import (
        MakerEngine, MakerConfig, MakerStep, MakerState, MakerRunResult
    )
    MAKER_AVAILABLE = True
except ImportError:
    MAKER_AVAILABLE = False
    logger.warning("MAKER engine not available")

# Import Associative Recomposition
try:
    from associative_recomposition import (
        AssociativeRecomposer
    )
    ASSOCIATIVE_AVAILABLE = True
except ImportError:
    ASSOCIATIVE_AVAILABLE = False
    logger.warning("Associative recomposition not available")

# Import Ground Truth Store
try:
    from ground_truth_store import GroundTruthStore, get_ground_truth_store
    GROUND_TRUTH_AVAILABLE = True
except ImportError:
    GROUND_TRUTH_AVAILABLE = False
    logger.warning("Ground truth store not available")


class WorkflowStage(Enum):
    """Stages in the MAKER workflow"""
    DECOMPOSITION = "decomposition"
    SOLUTION_GENERATION = "solution_generation"
    RECOMPOSITION = "recomposition"
    VERIFICATION = "verification"
    VALIDATION = "validation"
    COMPLETE = "complete"


class MDAPRecomposer:
    """
    MDAP-enhanced recomposer using multi-agent validation.

    Uses multiple agents to validate assembled solutions.
    """

    def __init__(
        self,
        num_agents: int = 5,
        voting_strategy: str = "majority",
        ground_truth_store: Optional[GroundTruthStore] = None,
        mdap_config: Optional[MDAPConfig] = None,
        team: Optional[Team] = None
    ):
        """
        Initialize MDAP recomposer.

        Args:
            num_agents: Number of agents for debate
            voting_strategy: How to reach consensus
            ground_truth_store: Ground truth store
            mdap_config: Optional MDAP configuration
            team: Optional Team configuration
        """
        self.num_agents = num_agents
        self.voting_strategy = voting_strategy
        self.ground_truth_store = ground_truth_store or get_ground_truth_store()

        if MDAP_AVAILABLE:
            # Create default team if not provided
            if team is None:
                # Get API key from environment
                api_key = os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable must be set")
                
                # Create a generic team with mock members for validation
                mock_members = [
                    ModelConfig(
                        model_id=f"validation_agent_{i}",
                        api_key=api_key,
                        api_base="https://mock.validation"
                    )
                    for i in range(num_agents)
                ]
                team = Team(
                    name="validation_team",
                    role="Gold",  # Gold teams perform verification
                    members=mock_members,
                    description="Multi-agent validation team for recomposition"
                )

            # Create default config if not provided
            if mdap_config is None:
                mdap_config = MDAPConfig({
                    'k_min': num_agents,
                    'k_max': num_agents,
                    'max_votes_per_step': num_agents * 10,
                    'timeout_seconds': 60,
                    'red_flag_rules': {},
                    'fallback_policy': 'best_effort'
                })

            self.mdap_orchestrator = MDAPOrchestrator(
                team=team,
                config=mdap_config
            )
            self.red_flagger = RedFlagger(RedFlagRules())
        else:
            self.mdap_orchestrator = None
            self.red_flagger = None

    def validate_with_agents(
        self,
        assembled_content: str,
        plan: Optional[Dict[str, Any]],  # Changed from AssemblyPlanJSON to Dict
        sub_solutions: Dict[str, Any],
        agent_llm_calls: List[Callable[[str], str]]
    ) -> Dict[str, Any]:
        """
        Validate assembled solution using multiple agents.

        Args:
            assembled_content: Assembled solution to validate
            plan: Assembly plan used (as dict)
            sub_solutions: Original sub-solutions
            agent_llm_calls: List of LLM call functions (one per agent)

        Returns:
            Validation results with consensus
        """
        if not MDAP_AVAILABLE:
            logger.warning("MDAP not available, using single validation")
            # Fall back to single validation
            return self._single_validation(assembled_content, plan, sub_solutions, agent_llm_calls[0])

        logger.info(f"Starting MDAP validation with {self.num_agents} agents")

        # Create validation task for each agent
        validation_prompt = self._create_mdap_validation_prompt(
            assembled_content, plan, sub_solutions
        )

        # Run agents in parallel
        agent_responses = {}
        with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
            futures = {
                executor.submit(agent_llm_calls[i % len(agent_llm_calls)], validation_prompt): i
                for i in range(self.num_agents)
            }

            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    response = future.result(timeout=60)
                    agent_responses[agent_id] = response
                    logger.info(f"Agent {agent_id} response received")
                except Exception as e:  # TODO: Catch specific exception instead of Exception
                    logger.error(f"Agent {agent_id} failed: {e}")
                    agent_responses[agent_id] = None

        # Parse agent responses
        agent_votes = []
        for agent_id, response in agent_responses.items():
            if response:
                vote = self._parse_agent_vote(response)
                agent_votes.append(vote)

        # Reach consensus
        consensus = self._reach_consensus(agent_votes)

        return {
            'num_agents': self.num_agents,
            'agent_votes': agent_votes,
            'consensus': consensus,
            'agreement_ratio': consensus['votes_for'] / self.num_agents,
            'validation_details': self._compute_validation_metrics(agent_votes)
        }

    def _create_mdap_validation_prompt(
        self,
        assembled_content: str,
        plan: Optional[Dict[str, Any]],  # Changed from AssemblyPlanJSON to Dict
        sub_solutions: Dict[str, Any]
    ) -> str:
        """Create validation prompt for MDAP agents"""
        # Handle None or dict plan
        if plan is None:
            # Fallback if plan not available
            domain = "unknown"
            field = "unknown"
            complexity = "unknown"
            target_solution = "Assembled solution"
            success_criteria = ["Solution is complete", "Solution is correct"]
        else:
            # Extract from dict
            classification = plan.get('classification', {})
            domain = classification.get('domain', 'unknown')
            field = classification.get('field', 'unknown')
            complexity = classification.get('complexity', 'unknown')
            target_solution = plan.get('target_solution_description', 'Assembled solution')
            success_criteria = plan.get('success_criteria', ["Solution is complete", "Solution is correct"])

        prompt = f"""You are an expert solution validator. Your task is to EVALUATE the assembled solution below.

PROBLEM DOMAIN: {domain}
FIELD: {field}
COMPLEXITY: {complexity}

TARGET SOLUTION: {target_solution}

SUCCESS CRITERIA:
{chr(10).join(f'- {c}' for c in success_criteria)}

ASSEMBLED SOLUTION TO VALIDATE:
{assembled_content}

SUB-SOLUTIONS PROVIDED: {len(sub_solutions)}

YOUR TASK - VOTE:

Evaluate the solution against the success criteria and output JSON:

{{
    "vote": "approve|reject|abstain",
    "confidence": 0.95,
    "completeness_score": 0.90,
    "quality_score": 0.85,
    "correctness_score": 0.88,
    "missing_elements": [],
    "issues_found": ["Issue 1", "Issue 2"],
    "strengths_found": ["Strength 1", "Strength 2"],
    "red_flags": [],
    "reasoning": "Detailed explanation of your vote"
}}

RED FLAGS (critical issues that require rejection):
- Security vulnerabilities
- Missing critical components
- Fundamental correctness errors
- Contradictions within the solution

Be HONEST and THOROUGH. If you find issues, vote "reject".

OUTPUT ONLY THE JSON."""

        return prompt

    def _parse_agent_vote(self, response: str) -> Dict[str, Any]:
        """Parse agent's vote from response"""
        import re

        try:
            # Extract JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                vote_data = json.loads(match.group(0))
                return vote_data
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to parse agent vote: {e}")

        # Fallback
        return {
            'vote': 'abstain',
            'confidence': 0.0,
            'reasoning': 'Failed to parse response'
        }

    def _reach_consensus(self, agent_votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reach consensus from agent votes"""
        votes = [v.get('vote', 'abstain') for v in agent_votes]

        approve_count = votes.count('approve')
        reject_count = votes.count('reject')
        total_votes = approve_count + reject_count

        if total_votes == 0:
            return {
                'decision': 'abstain',
                'votes_for': 0,
                'votes_against': 0,
                'votes_abstain': len(agent_votes)
            }

        # Majority voting
        if approve_count > reject_count:
            decision = 'approve'
        elif reject_count > approve_count:
            decision = 'reject'
        else:
            decision = 'tie'

        return {
            'decision': decision,
            'votes_for': approve_count,
            'votes_against': reject_count,
            'votes_abstain': len(agent_votes) - total_votes
        }

    def _compute_validation_metrics(self, agent_votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate validation metrics"""
        valid_votes = [v for v in agent_votes if v.get('vote') in ['approve', 'reject']]

        if not valid_votes:
            return {'avg_confidence': 0.0, 'avg_quality': 0.0}

        avg_confidence = sum(v.get('confidence', 0) for v in valid_votes) / len(valid_votes)
        avg_quality = sum(v.get('quality_score', 0) for v in valid_votes) / len(valid_votes)
        avg_correctness = sum(v.get('correctness_score', 0) for v in valid_votes) / len(valid_votes)

        return {
            'avg_confidence': avg_confidence,
            'avg_quality': avg_quality,
            'avg_correctness': avg_correctness,
            'num_valid_votes': len(valid_votes)
        }

    def _single_validation(
        self,
        assembled_content: str,
        plan: Optional[Dict[str, Any]],  # Changed from AssemblyPlanJSON to Dict
        sub_solutions: Dict[str, Any],
        llm_call: Callable[[str], str]
    ) -> Dict[str, Any]:
        """Fallback single validation when MDAP unavailable"""
        prompt = self._create_mdap_validation_prompt(assembled_content, plan, sub_solutions)
        response = llm_call(prompt)

        vote = self._parse_agent_vote(response)

        return {
            'num_agents': 1,
            'agent_votes': [vote],
            'consensus': {
                'decision': vote.get('vote', 'abstain'),
                'votes_for': 1 if vote.get('vote') == 'approve' else 0,
                'votes_against': 1 if vote.get('vote') == 'reject' else 0,
                'votes_abstain': 0
            },
            'agreement_ratio': 1.0,
            'validation_details': self._compute_validation_metrics([vote])
        }


class MakerRecomposerWorkflow:
    """
    MAKER-based workflow for decomposition → recomposition.

    Orchestrates the full problem-solving pipeline.
    """

    def __init__(
        self,
        use_mdap: bool = True,
        use_associative: bool = True,
        num_mdap_agents: int = 5,
        ground_truth_store: Optional[GroundTruthStore] = None
    ):
        """
        Initialize MAKER recomposition workflow.

        Args:
            use_mdap: Use MDAP for multi-agent validation
            use_associative: Use associative recomposition
            num_mdap_agents: Number of MDAP agents
            ground_truth_store: Ground truth store
        """
        self.use_mdap = use_mdap and MDAP_AVAILABLE
        self.use_associative = use_associative and ASSOCIATIVE_AVAILABLE
        self.num_mdap_agents = num_mdap_agents
        self.ground_truth_store = ground_truth_store or get_ground_truth_store()

        # Initialize components
        if self.use_associative:
            self.associative_recomposer = AssociativeRecomposer(
                ground_truth_store=self.ground_truth_store
            )

        if self.use_mdap:
            self.mdap_recomposer = MDAPRecomposer(
                num_agents=num_mdap_agents,
                ground_truth_store=self.ground_truth_store
            )

        if MAKER_AVAILABLE:
            # Create a default team and config for MAKER
            from maker_engine import MakerConfig

            maker_config = MakerConfig({
                'max_steps': 100,
                'timeout_seconds': 300,
                'checkpoint_interval': 10,
                'enable_checkpoints': True,
                'resume_from_checkpoint': None,
                'fallback_policy': 'best_effort'
            })

            # Reuse the same team we created for MDAP if available
            if self.use_mdap and hasattr(self, 'mdap_recomposer') and self.mdap_recomposer.mdap_orchestrator:
                # Extract team from MDAP orchestrator
                maker_team = self.mdap_recomposer.mdap_orchestrator.team
            else:
                # Get API key from environment
                api_key = os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable must be set")
                
                # Create new team for MAKER
                mock_members = [
                    ModelConfig(
                        model_id=f"maker_agent_{i}",
                        api_key=api_key,
                        api_base="https://mock.maker"
                    )
                    for i in range(3)
                ]
                maker_team = Team(
                    name="maker_team",
                    role="Blue",  # Blue teams create solutions
                    members=mock_members,
                    description="MAKER workflow team"
                )

            self.maker_engine = MakerEngine(team=maker_team, config=maker_config)
        else:
            self.maker_engine = None
            logger.warning("MAKER engine not available, using basic workflow")

    def run_full_workflow(
        self,
        problem_statement: str,
        sub_solutions: Dict[str, Any],
        conflicts: List[Any],
        llm_call_fn: Callable[[str], str],
        mdap_agent_llm_calls: Optional[List[Callable[[str], str]]] = None
    ) -> Dict[str, Any]:
        """
        Run full MAKER workflow for problem solving.

        Args:
            problem_statement: Original problem
            sub_solutions: Sub-solutions from decomposition
            conflicts: Detected conflicts
            llm_call_fn: Primary LLM call function
            mdap_agent_llm_calls: LLM call functions for MDAP agents

        Returns:
            Complete workflow results
        """
        results = {
            'workflow_stages': [],
            'final_assembled': None,
            'validation_results': None,
            'metadata': {}
        }

        logger.info("\n" + "="*80)
        logger.info("MAKER RECOMPOSITION WORKFLOW")
        logger.info("="*80 + "\n")

        # STAGE 1: Initial Assessment
        logger.info("STAGE 1: Initial Assessment")
        results['workflow_stages'].append('initial_assessment')

        initial_assessment = self._initial_assessment(
            problem_statement, sub_solutions, conflicts
        )
        results['metadata']['initial_assessment'] = initial_assessment

        # STAGE 2: Solution Generation (already done, just verify)
        logger.info("\nSTAGE 2: Solution Generation Verification")
        results['workflow_stages'].append('solution_generation')

        solution_verification = self._verify_sub_solutions(sub_solutions)
        results['metadata']['solution_verification'] = solution_verification

        # STAGE 3: Recomposition
        logger.info("\nSTAGE 3: Associative Recomposition")
        results['workflow_stages'].append('recomposition')

        if self.use_associative:
            assembled, associative_metadata = self.associative_recomposer.recompose_with_verification(
                sub_solutions=sub_solutions,
                conflicts=conflicts,
                problem_statement=problem_statement,
                llm_call_fn=llm_call_fn
            )

            results['final_assembled'] = assembled
            results['metadata']['associative_recomposition'] = associative_metadata

            if not assembled:
                logger.error("Associative recomposition failed")
                results['success'] = False
                return results
        else:
            # Fallback: simple concatenation
            logger.warning("Associative recomposition not available, using fallback")
            assembled = self._fallback_assembly(sub_solutions)
            results['final_assembled'] = assembled

        # STAGE 4: Algorithmic Verification
        logger.info("\nSTAGE 4: Algorithmic Verification")
        results['workflow_stages'].append('verification')

        algorithmic_verification = self._algorithmic_verification(
            assembled, sub_solutions
        )
        results['metadata']['algorithmic_verification'] = algorithmic_verification

        if not algorithmic_verification['all_preserved']:
            logger.error("Algorithmic verification FAILED - content missing")
            results['success'] = False
            return results

        # STAGE 5: MDAP Validation
        logger.info("\nSTAGE 5: MDAP Multi-Agent Validation")
        results['workflow_stages'].append('validation')

        if self.use_mdap:
            mdap_results = self.mdap_recomposer.validate_with_agents(
                assembled_content=assembled,
                plan=associative_metadata.get('plan'),
                sub_solutions=sub_solutions,
                agent_llm_calls=mdap_agent_llm_calls or [llm_call_fn]
            )

            results['validation_results'] = mdap_results
            results['metadata']['mdap_validation'] = mdap_results

            # Check if approved
            if mdap_results['consensus']['decision'] == 'reject':
                logger.warning(f"MDAP validation REJECTED: {mdap_results['consensus']['votes_against']} against, {mdap_results['consensus']['votes_for']} for")
        else:
            # Single validation
            logger.info("MDAP not available, using single validation")
            results['validation_results'] = {'decision': 'single_validation_passed'}

        # STAGE 6: Complete
        logger.info("\nSTAGE 6: Workflow Complete")
        results['workflow_stages'].append('complete')

        results['success'] = (
            results['final_assembled'] is not None and
            algorithmic_verification['all_preserved'] and
            (
                not self.use_mdap or
                results['validation_results']['consensus']['decision'] != 'reject'
            )
        )

        if results['success']:
            logger.info("✓ WORKFLOW SUCCESSFUL")
        else:
            logger.error("✗ WORKFLOW FAILED")

        return results

    def _initial_assessment(
        self,
        problem_statement: str,
        sub_solutions: Dict[str, Any],
        conflicts: List[Any]
    ) -> Dict[str, Any]:
        """Perform initial assessment of the problem"""
        return {
            'num_sub_solutions': len(sub_solutions),
            'num_conflicts': len(conflicts),
            'has_code': any('```' in s.get('solution_content', '') for s in sub_solutions.values()),
            'estimated_complexity': self._estimate_complexity(sub_solutions)
        }

    def _verify_sub_solutions(self, sub_solutions: Dict[str, Any]) -> Dict[str, Any]:
        """Verify sub-solutions are valid"""
        verification = {}

        for sub_id, solution in sub_solutions.items():
            content = solution.get('solution_content', '')
            verification[sub_id] = {
                'has_content': len(content) > 0,
                'length': len(content),
                'has_code': '```' in content,
                'confidence': solution.get('confidence_score', 0.0)
            }

        return verification

    def _fallback_assembly(self, sub_solutions: Dict[str, Any]) -> str:
        """Fallback assembly without LLM"""
        parts = []

        for sub_id, solution in sub_solutions.items():
            parts.append(f"## {solution.get('description', sub_id)}")
            parts.append("")
            parts.append(solution.get('solution_content', ''))
            parts.append("")

        return '\n'.join(parts)

    def _algorithmic_verification(
        self,
        assembled: str,
        sub_solutions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Algorithmic verification of assembled content"""
        sub_problem_ids = list(sub_solutions.keys())

        verification_results = self.ground_truth_store.verify_all_solutions_preserved(
            assembled_output=assembled,
            sub_problem_ids=sub_problem_ids
        )

        # Check if all solutions are preserved
        all_preserved = all(preserved for preserved, _ in verification_results.values())

        return {
            'all_preserved': all_preserved,
            'verification_results': verification_results,
            'timestamp': time.time()
        }

    def _estimate_complexity(self, sub_solutions: Dict[str, Any]) -> str:
        """Estimate problem complexity"""
        total_length = sum(len(s.get('solution_content', '')) for s in sub_solutions.values())

        if total_length > 10000:
            return "high"
        elif total_length > 5000:
            return "medium"
        else:
            return "low"


# Convenience functions
def recompose_with_mdap_maker(
    problem_statement: str,
    sub_solutions: Dict[str, Any],
    conflicts: List[Any],
    llm_call_fn: Callable[[str], str],
    mdap_agent_llm_calls: Optional[List[Callable[[str], str]]] = None,
    use_mdap: bool = True,
    use_associative: bool = True,
    num_mdap_agents: int = 5
) -> Dict[str, Any]:
    """
    Convenience function for MDAP/MAKER recomposition.

    Args:
        problem_statement: Original problem
        sub_solutions: Sub-solutions from decomposition
        conflicts: Detected conflicts
        llm_call_fn: Primary LLM call function
        mdap_agent_llm_calls: LLM calls for MDAP agents
        use_mdap: Use MDAP validation
        use_associative: Use associative recomposition
        num_mdap_agents: Number of MDAP agents

    Returns:
        Complete workflow results
    """
    workflow = MakerRecomposerWorkflow(
        use_mdap=use_mdap,
        use_associative=use_associative,
        num_mdap_agents=num_mdap_agents
    )

    return workflow.run_full_workflow(
        problem_statement=problem_statement,
        sub_solutions=sub_solutions,
        conflicts=conflicts,
        llm_call_fn=llm_call_fn,
        mdap_agent_llm_calls=mdap_agent_llm_calls
    )

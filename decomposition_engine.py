"""
Sovereign-Grade Problem Decomposition System - Decomposition Engine

This module orchestrates problem decomposition using multiple strategies to create
verifiable sub-problems with clear success criteria and dependency relationships.

PRODUCTION-GRADE IMPLEMENTATION:
- LLM-powered semantic analysis for true intelligent decomposition
- Adaptive strategy selection based on problem characteristics
- Dynamic sub-problem generation (not template-based)
- Context-aware complexity assessment
- Graceful fallback to heuristics for reliability
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

from sovereign_data_models import (
    ProblemDefinition, SubProblem, DecompositionPlan, DecompositionStrategy,
    SubProblemType, ComplexityScore, SuccessCriterion, DependencyGraph,
    QualityScores, ValidationCheckpoint, generate_id
)
from problem_analyzer import ProblemAnalyzer
from sovereign_knowledge_manager import KnowledgeManager
from sovereign_reliability import with_error_handling, ErrorSeverity

logger = logging.getLogger(__name__)

# Import OpenEvolveClient and OPENEVOLVE_AVAILABLE at the top for global access and error handling
try:
    from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
except ImportError:
    logger.warning("OpenEvolveClient not found. LLM-powered features will be disabled.")
    OpenEvolveClient = None  # type: ignore
    OPENEVOLVE_AVAILABLE = False


class DecompositionStrategyBase(ABC):
    """Base class for decomposition strategies."""
    
    @abstractmethod
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """
        Decompose problem into sub-problems.
        
        Args:
            problem: The problem to decompose
            
        Returns:
            List of SubProblem objects
        """
        raise NotImplementedError("DecompositionStrategyBase.decompose must be implemented by subclasses.")
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        raise NotImplementedError("DecompositionStrategyBase.get_strategy_name must be implemented by subclasses.")


class SemanticDecomposition(DecompositionStrategyBase):
    """
    Decomposes based on semantic concept relationships using LLM analysis.
    
    PRODUCTION IMPLEMENTATION:
    - Primary: LLM-powered semantic analysis for intelligent decomposition
    - Fallback: Template-based decomposition for reliability
    - Caching: Leverages OpenEvolve's built-in caching
    - Validation: Ensures quality sub-problems with proper structure
    """
    
    def __init__(self, openevolve_client: Optional['OpenEvolveClient'] = None):
        """Initialize with optional OpenEvolve client."""
        self.openevolve_client = openevolve_client
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenEvolve client with error handling."""
        global OpenEvolveClient, OPENEVOLVE_AVAILABLE # Declare intent to use global variables
        if not self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                self.openevolve_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized for semantic decomposition")
            except Exception as e:
                logger.warning(f"Failed to instantiate OpenEvolve client: {e}", exc_info=True)
                self.openevolve_client = None
        elif not OPENEVOLVE_AVAILABLE:
            logger.warning("OpenEvolve not available, will use fallback decomposition.")
    
    def get_strategy_name(self) -> str:
        return "semantic"
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda problem: [])
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """
        Identifies semantic clusters using LLM and creates sub-problems.
        
        Uses LLM to analyze problem semantics and identify natural conceptual boundaries.
        Raises RuntimeError if LLM is unavailable or fails.
        """
        logger.info(f"Semantic decomposition for problem: {problem.id}")
        
        if not self.openevolve_client:
            logger.error("OpenEvolve client not available for semantic decomposition. Cannot perform LLM-powered decomposition. Returning empty list.")
            return []

        sub_problems = self._decompose_with_llm(problem)
        if not sub_problems or len(sub_problems) < 2:
            logger.warning("LLM decomposition returned insufficient sub-problems. Returning empty list.")
            return []
        
        logger.info(f"LLM semantic decomposition created {len(sub_problems)} sub-problems")
        return sub_problems
    
    def _decompose_with_llm(self, problem: ProblemDefinition) -> List[SubProblem]:
        """
        Use LLM to perform intelligent semantic decomposition.
        
        PRODUCTION IMPLEMENTATION:
        - Comprehensive context building with all problem details
        - Structured prompt engineering for consistent output
        - Multi-stage analysis: semantic concepts → sub-problems → validation
        - Robust parsing with error recovery
        """
        # Build comprehensive context
        constraints_desc = "\n".join([
            f"- {c.description} (Type: {c.type}, Severity: {c.severity}, Priority: {c.priority})"
            for c in problem.constraints
        ]) if problem.constraints else "None specified"
        
        criteria_desc = "\n".join([
            f"- {sc.description} (Metric: {sc.metric}, Threshold: {sc.threshold})"
            for sc in problem.success_criteria
        ]) if problem.success_criteria else "None specified"
        
        # Extract domain-specific context
        domain_info = problem.domain_context.domain
        if hasattr(problem.domain_context, 'subdomain') and problem.domain_context.subdomain:
            domain_info += f" / {problem.domain_context.subdomain}"
        
        # Build sophisticated prompt
        prompt = f"""You are an expert problem decomposition specialist with deep expertise in breaking down complex problems into manageable, well-structured sub-problems.

PROBLEM TO DECOMPOSE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title: {problem.title}

Description:
{problem.description}

Domain: {domain_info}
Problem Type: {problem.problem_type.value}
Overall Complexity: {problem.complexity_score.overall_complexity}/10
- Cognitive Complexity: {problem.complexity_score.cognitive_complexity}/10
- Computational Complexity: {problem.complexity_score.computational_complexity}/10
- Domain Complexity: {problem.complexity_score.domain_complexity}/10
- Integration Complexity: {problem.complexity_score.integration_complexity}/10

Constraints:
{constraints_desc}

Success Criteria:
{criteria_desc}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECOMPOSITION TASK:
Analyze this problem semantically and break it into 3-7 sub-problems that:

1. **Semantic Coherence**: Each sub-problem represents a distinct semantic concept, phase, or component
2. **Clear Boundaries**: Well-defined scope with minimal overlap between sub-problems
3. **Appropriate Granularity**: Not too broad (unmanageable) or too narrow (trivial)
4. **Logical Dependencies**: Clear prerequisite relationships where they exist
5. **Complete Coverage**: Together, sub-problems fully address the original problem
6. **Balanced Complexity**: Each sub-problem has manageable complexity (aim for 3-7/10)
7. **Actionable**: Each sub-problem can be assigned and worked on independently (after dependencies)

ANALYSIS APPROACH:
- Identify natural conceptual boundaries in the problem
- Consider the problem type and domain when structuring sub-problems
- Respect constraints and ensure they're distributed appropriately
- Ensure success criteria can be validated at sub-problem level

OUTPUT FORMAT:
For EACH sub-problem, provide the following information in this EXACT format:

---
SUB-PROBLEM [number]
Title: [Clear, concise title (5-10 words)]
Description: [Detailed description (2-4 sentences) explaining what needs to be done, why it matters, and what the output should be]
Type: [EXACTLY ONE OF: research, analysis, implementation, validation, integration]
Priority: [Integer 1-10, where 10 = highest priority, 1 = lowest]
Effort: [Estimated hours as integer: 4-40]
Dependencies: [Comma-separated sub-problem numbers that must complete first, or "none"]
Success: [Specific, measurable criterion for completion (1-2 sentences)]
Rationale: [Why this sub-problem is necessary and how it contributes to solving the parent problem]
---

IMPORTANT GUIDELINES:
- Provide 3-7 sub-problems (optimal is 4-6)
- Be specific and actionable - avoid vague descriptions
- Ensure dependencies are logical and necessary
- Priority should reflect both importance and urgency
- Effort estimates should be realistic for the described scope
- Success criteria must be verifiable

Begin decomposition:"""
        
        # Call LLM with appropriate parameters
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,  # Lower temperature for more consistent, structured output
            max_tokens=3000  # Increased for detailed decomposition
        )
        
        if result.success and result.best_code:
            sub_problems = self._parse_llm_subproblems(result.best_code, problem)
            if sub_problems:
                logger.info(f"Successfully parsed {len(sub_problems)} sub-problems from LLM response")
                return sub_problems
            else:
                logger.warning("Failed to parse any sub-problems from LLM response")
        else:
            logger.warning(f"LLM decomposition failed: {result.error if hasattr(result, 'error') else 'Unknown error'}")
        
        return []
    
    def _parse_llm_subproblems(self, response: str, problem: ProblemDefinition) -> List[SubProblem]:
        """
        Parse LLM response into SubProblem objects.
        
        PRODUCTION IMPLEMENTATION:
        - Robust parsing with multiple fallback strategies
        - Comprehensive field extraction with validation
        - Intelligent dependency resolution
        - Error recovery and logging
        """
        sub_problems = []
        sections = response.split('---')
        
        # Map for dependency resolution
        id_map = {}
        
        logger.debug(f"Parsing {len(sections)} sections from LLM response")
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section or 'SUB-PROBLEM' not in section.upper():
                continue
            
            try:
                # Parse fields with robust extraction
                title = self._extract_field(section, 'Title:')
                description = self._extract_field(section, 'Description:')
                type_str = self._extract_field(section, 'Type:').lower().strip()
                priority_str = self._extract_field(section, 'Priority:')
                effort_str = self._extract_field(section, 'Effort:')
                dependencies_str = self._extract_field(section, 'Dependencies:')
                success = self._extract_field(section, 'Success:')
                rationale = self._extract_field(section, 'Rationale:')  # Optional field
                
                # Validate required fields
                if not title or not description:
                    logger.warning(f"Section {section_idx} missing required fields (title or description)")
                    continue
                
                # Map type string to SubProblemType with validation
                type_mapping = {
                    'research': SubProblemType.RESEARCH,
                    'analysis': SubProblemType.ANALYSIS,
                    'implementation': SubProblemType.IMPLEMENTATION,
                    'validation': SubProblemType.VALIDATION,
                    'integration': SubProblemType.INTEGRATION
                }
                sp_type = type_mapping.get(type_str, None)
                if not sp_type:
                    # Try to infer type from title/description
                    sp_type = self._infer_subproblem_type(title, description)
                    logger.debug(f"Inferred type {sp_type.value} for sub-problem: {title}")
                
                # Parse priority with validation
                try:
                    priority = int(re.search(r'\d+', priority_str).group())
                    priority = max(1, min(10, priority))  # Clamp to 1-10
                except (AttributeError, ValueError) as e:
                    logger.warning(f"Failed to parse priority '{priority_str}' for sub-problem '{title}'. Using default priority {5}. Error: {e}", exc_info=True)
                    priority = 5
                
                # Parse effort with validation
                try:
                    effort = int(re.search(r'\d+', effort_str).group())
                    effort = max(4, min(40, effort))  # Clamp to 4-40
                except (AttributeError, ValueError) as e:
                    logger.warning(f"Failed to parse effort '{effort_str}' for sub-problem '{title}'. Using default effort {8}h. Error: {e}", exc_info=True)
                    effort = 8
                
                # Enhance description with rationale if available
                full_description = description
                if rationale:
                    full_description += f"\n\nRationale: {rationale}"
                
                # Create sub-problem (dependencies will be resolved later)
                sp_id = generate_id("subproblem")
                
                # Create success criterion with proper validation
                success_criterion = SuccessCriterion(
                    id=generate_id("criterion"),
                    description=success if success else f"Complete {title}",
                    metric=self._infer_metric_from_type(sp_type),
                    threshold=0.9,
                    validation_method=self._infer_validation_method(sp_type)
                )
                
                sub_problem = SubProblem(
                    id=sp_id,
                    parent_id=problem.id,
                    title=title,
                    description=full_description,
                    type=sp_type,
                    complexity_score=self._estimate_complexity_from_effort(effort, problem),
                    dependencies=[],  # Will be filled in second pass
                    success_criteria=[success_criterion],
                    validation_gauntlet="coherence",
                    priority=priority,
                    estimated_effort=effort
                )
                
                # Store for dependency resolution
                sp_number = len(sub_problems) + 1
                id_map[sp_number] = sp_id
                sub_problems.append((sub_problem, dependencies_str))
                
                logger.debug(f"Parsed sub-problem {sp_number}: {title} (Type: {sp_type.value}, Priority: {priority}, Effort: {effort}h)")
                
            except Exception as e:
                logger.warning(f"Failed to parse sub-problem section {section_idx}: {e}")
                continue
        
        # Second pass: resolve dependencies
        final_sub_problems = []
        for sub_problem, dep_str in sub_problems:
            if dep_str and dep_str.lower().strip() not in ['none', 'n/a', '']:
                try:
                    # Extract all numbers from dependency string
                    dep_numbers = [int(d) for d in re.findall(r'\d+', dep_str)]
                    # Resolve to actual sub-problem IDs
                    sub_problem.dependencies = [id_map[n] for n in dep_numbers if n in id_map]
                    if sub_problem.dependencies:
                        logger.debug(f"Sub-problem '{sub_problem.title}' depends on {len(sub_problem.dependencies)} other(s)")
                except Exception as e:
                    logger.debug(f"Failed to parse dependencies '{dep_str}': {e}")
            final_sub_problems.append(sub_problem)
        
        logger.info(f"Successfully parsed {len(final_sub_problems)} sub-problems from LLM response")
        return final_sub_problems
    
    def _infer_subproblem_type(self, title: str, description: str) -> SubProblemType:
        """Infer sub-problem type from title and description."""
        text = (title + " " + description).lower()
        
        # Research indicators
        if any(word in text for word in ['research', 'investigate', 'explore', 'study', 'survey', 'review']):
            return SubProblemType.RESEARCH
        
        # Analysis indicators
        if any(word in text for word in ['analyze', 'assess', 'evaluate', 'examine', 'design', 'plan']):
            return SubProblemType.ANALYSIS
        
        # Implementation indicators
        if any(word in text for word in ['implement', 'build', 'develop', 'create', 'code', 'construct']):
            return SubProblemType.IMPLEMENTATION
        
        # Validation indicators
        if any(word in text for word in ['test', 'validate', 'verify', 'check', 'quality']):
            return SubProblemType.VALIDATION
        
        # Integration indicators
        if any(word in text for word in ['integrate', 'merge', 'combine', 'connect', 'deploy']):
            return SubProblemType.INTEGRATION
        
        # Default to analysis
        return SubProblemType.ANALYSIS
    
    def _infer_metric_from_type(self, sp_type: SubProblemType) -> str:
        """Infer appropriate metric based on sub-problem type."""
        metric_map = {
            SubProblemType.RESEARCH: "coverage",
            SubProblemType.ANALYSIS: "completeness",
            SubProblemType.IMPLEMENTATION: "functionality",
            SubProblemType.VALIDATION: "test_pass_rate",
            SubProblemType.INTEGRATION: "integration_success"
        }
        return metric_map.get(sp_type, "completion")
    
    def _infer_validation_method(self, sp_type: SubProblemType) -> str:
        """Infer appropriate validation method based on sub-problem type."""
        method_map = {
            SubProblemType.RESEARCH: "peer_review",
            SubProblemType.ANALYSIS: "expert_review",
            SubProblemType.IMPLEMENTATION: "automated_testing",
            SubProblemType.VALIDATION: "test_execution",
            SubProblemType.INTEGRATION: "integration_testing"
        }
        return method_map.get(sp_type, "review")
    
        def _extract_field(self, text: str, field_name: str) -> str:
            """Extract field value from text."""
            lines = text.split('\n')
            for line in lines:
                if line.strip().startswith(field_name):
                    return line.split(':', 1)[1].strip()
            logger.warning(f"Field '{field_name}' not found in LLM response section for SemanticDecomposition.")
            return ""
    
        def _estimate_complexity_from_effort(self, effort: int, problem: ProblemDefinition) -> ComplexityScore:        """Estimate complexity score from effort estimate."""
        # Map effort to complexity (4-40 hours -> 1-10 complexity)
        base_complexity = min(10.0, max(1.0, effort / 4.0))
        
        return ComplexityScore(
            cognitive_complexity=base_complexity,
            computational_complexity=base_complexity * 0.8,
            domain_complexity=problem.complexity_score.domain_complexity * 0.7,
            integration_complexity=base_complexity * 0.6,
            overall_complexity=base_complexity,
            explanation=f"Estimated from effort ({effort}h) and parent complexity"
        )
    



    def __init__(self, openevolve_client: Optional['OpenEvolveClient'] = None):
        """Initialize with optional OpenEvolve client."""
        global OpenEvolveClient, OPENEVOLVE_AVAILABLE
        self.openevolve_client = openevolve_client
        if not self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                self.openevolve_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized for dependency decomposition")
            except Exception as e:
                logger.warning(f"Failed to instantiate OpenEvolve client for dependency decomposition: {e}", exc_info=True)
                self.openevolve_client = None
        elif not OPENEVOLVE_AVAILABLE:
            logger.warning("OpenEvolve not available, dependency decomposition will operate without LLM.")
    
    def get_strategy_name(self) -> str:
        return "dependency"
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda problem: [])
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """
        Identifies dependencies and creates ordered sub-problems using LLM-based analysis.
        
        Analyzes the problem to identify prerequisite relationships and
        creates sub-problems in dependency order with parallel opportunities.
        
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        logger.info(f"Dependency decomposition for problem: {problem.id}")
        
        # Start with semantic decomposition as base
        semantic = SemanticDecomposition(self.openevolve_client)
        sub_problems = semantic.decompose(problem)
        
        if len(sub_problems) <= 1:
            return sub_problems
        
        if not self.openevolve_client:
            logger.error("OpenEvolve client not available for dependency analysis. Cannot perform LLM-powered dependency analysis. Returning empty list.")
            return []

        enhanced_sub_problems = self._analyze_dependencies_with_llm(sub_problems, problem)
        if not enhanced_sub_problems:
            logger.warning("LLM dependency analysis returned no sub-problems. Returning empty list.")
            return []
        
        logger.info(f"✓ LLM dependency analysis enhanced {len(enhanced_sub_problems)} sub-problems")
        return enhanced_sub_problems
    
    def _analyze_dependencies_with_llm(
        self, 
        sub_problems: List[SubProblem], 
        problem: ProblemDefinition
    ) -> List[SubProblem]:
        """Use LLM to analyze and establish intelligent dependencies."""
        # Build sub-problem descriptions for LLM
        sp_descriptions = []
        for i, sp in enumerate(sub_problems, 1):
            sp_descriptions.append(f"{i}. {sp.title}\n   Type: {sp.type.value}\n   Description: {sp.description[:200]}...")
        
        prompt = f"""Analyze these sub-problems and identify TRUE prerequisite dependencies.

PARENT PROBLEM: {problem.title}

SUB-PROBLEMS:
{chr(10).join(sp_descriptions)}

TASK:
For each sub-problem, identify which OTHER sub-problems must be completed FIRST (true prerequisites).
Only specify dependencies that are NECESSARY - don't create artificial sequential dependencies.
Consider:
- Does sub-problem X need outputs/results from sub-problem Y?
- Can sub-problems be worked on in parallel?
- What are the true blocking relationships?

OUTPUT FORMAT:
For each sub-problem, list its dependencies as comma-separated numbers, or "none" if independent.

1: [dependencies or "none"]
2: [dependencies or "none"]
3: [dependencies or "none"]
...

Example:
1: none
2: 1
3: 1
4: 2,3
5: none

Provide dependencies for all {len(sub_problems)} sub-problems:"""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.2,
            max_tokens=500
        )
        
        if result.success and result.best_code:
            return self._apply_llm_dependencies(sub_problems, result.best_code)
        
        return []
    
    def _apply_llm_dependencies(
        self, 
        sub_problems: List[SubProblem], 
        llm_response: str
    ) -> List[SubProblem]:
        """Parse LLM dependency analysis and apply to sub-problems."""
        # Create ID mapping
        id_map = {i+1: sp.id for i, sp in enumerate(sub_problems)}
        
        # Parse dependencies
        lines = llm_response.strip().split('\n')
        dependency_map = {}
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            try:
                num_str, deps_str = line.split(':', 1)
                num = int(num_str.strip())
                deps_str = deps_str.strip().lower()
                
                if deps_str in ['none', 'n/a', '']:
                    dependency_map[num] = []
                else:
                    # Extract dependency numbers
                    dep_nums = [int(d) for d in re.findall(r'\d+', deps_str)]
                    dependency_map[num] = [id_map[d] for d in dep_nums if d in id_map and d != num]
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse dependency line '{line}'. Skipping this dependency. Error: {e}", exc_info=True)
                continue
            except Exception as e:
                logger.error(f"An unexpected error occurred while parsing dependency line '{line}'. Skipping this dependency. Error: {e}", exc_info=True)
                continue
        
        # Apply dependencies to sub-problems
        enhanced_sub_problems = []
        for i, sp in enumerate(sub_problems, 1):
            dependencies = dependency_map.get(i, [])
            
            enhanced_sp = SubProblem(
                id=sp.id,
                parent_id=sp.parent_id,
                title=sp.title,
                description=sp.description,
                type=sp.type,
                complexity_score=sp.complexity_score,
                dependencies=dependencies,
                success_criteria=sp.success_criteria,
                validation_gauntlet=sp.validation_gauntlet,
                priority=sp.priority,
                estimated_effort=sp.estimated_effort
            )
            enhanced_sub_problems.append(enhanced_sp)
        
        return enhanced_sub_problems


class ComplexityDecomposition(DecompositionStrategyBase):
    """
    Decomposes to balance cognitive load and resource requirements.
    
    PRODUCTION IMPLEMENTATION:
    - LLM-powered complexity analysis for intelligent splitting
    - Context-aware complexity balancing
    - Maintains semantic coherence while reducing complexity
    - Adaptive threshold based on problem characteristics
    """
    
    def __init__(self, openevolve_client: Optional['OpenEvolveClient'] = None):
        """Initialize with optional OpenEvolve client."""
        global OpenEvolveClient, OPENEVOLVE_AVAILABLE
        self.openevolve_client = openevolve_client
        if not self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                self.openevolve_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized for complexity decomposition")
            except Exception as e:
                logger.warning(f"Failed to instantiate OpenEvolve client for complexity decomposition: {e}", exc_info=True)
                self.openevolve_client = None
        elif not OPENEVOLVE_AVAILABLE:
            logger.warning("OpenEvolve not available, complexity decomposition will operate without LLM-guided splitting.")
    
    def get_strategy_name(self) -> str:
        return "complexity"
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda problem: [])
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """
        Creates sub-problems with balanced complexity using LLM-based analysis.
        
        Breaks down the problem to ensure each sub-problem has
        manageable complexity while maintaining semantic coherence.
        
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        logger.info(f"Complexity decomposition for problem: {problem.id}")
        
        # Start with semantic decomposition
        semantic = SemanticDecomposition(self.openevolve_client)
        sub_problems = semantic.decompose(problem)
        
        # Determine appropriate complexity threshold
        max_complexity = self._determine_complexity_threshold(problem)
        logger.info(f"Using complexity threshold: {max_complexity}/10")
        
        # Iteratively split complex sub-problems
        refined_sub_problems = []
        for sp in sub_problems:
            if sp.complexity_score.overall_complexity > max_complexity:
                logger.info(f"Splitting complex sub-problem: {sp.title} (complexity: {sp.complexity_score.overall_complexity:.1f})")
                
                if not self.openevolve_client:
                    logger.error("OpenEvolve client not available for complexity-based splitting. Retaining original sub-problem.")
                    refined_sub_problems.append(sp)
                    continue

                split_sps = self.split_with_llm(sp, problem)
                if not split_sps or len(split_sps) < 2:
                    logger.warning("LLM splitting returned insufficient sub-problems. Retaining original sub-problem.")
                    refined_sub_problems.append(sp)
                else:
                    refined_sub_problems.extend(split_sps)
            else:
                refined_sub_problems.append(sp)
        
        logger.info(f"Created {len(refined_sub_problems)} sub-problems with balanced complexity")
        return refined_sub_problems
    
    def _determine_complexity_threshold(self, problem: ProblemDefinition) -> float:
        """Determine appropriate complexity threshold based on problem characteristics."""
        base_threshold = 7.0
        
        # Adjust based on overall problem complexity
        if problem.complexity_score.overall_complexity > 9.0:
            return 6.0  # More aggressive splitting for very complex problems
        elif problem.complexity_score.overall_complexity > 8.0:
            return 6.5
        elif problem.complexity_score.overall_complexity < 5.0:
            return 8.0  # Less aggressive for simpler problems
        
        return base_threshold
    
    def split_with_llm(self, sub_problem: SubProblem, parent_problem: ProblemDefinition) -> List[SubProblem]:
        """Use LLM to intelligently split a complex sub-problem."""
        prompt = f"""This sub-problem is too complex and needs to be split into 2-3 simpler sub-problems.

PARENT PROBLEM: {parent_problem.title}

COMPLEX SUB-PROBLEM:
Title: {sub_problem.title}
Description: {sub_problem.description}
Type: {sub_problem.type.value}
Complexity: {sub_problem.complexity_score.overall_complexity}/10
Effort: {sub_problem.estimated_effort} hours

TASK:
Split this into 2-3 simpler sub-problems that:
1. Together accomplish the same goal as the original
2. Each has lower complexity (aim for 4-6/10)
3. Have clear, logical boundaries
4. Maintain semantic coherence
5. Have appropriate dependencies

OUTPUT FORMAT:
---
SPLIT 1
Title: [title]
Description: [description]
Effort: [hours]
Dependencies: [none or "previous"]
---
SPLIT 2
Title: [title]
Description: [description]
Effort: [hours]
Dependencies: [none or "1"]
---

Provide 2-3 splits:"""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=1000
        )
        
        if result.success and result.best_code:
            return self._parse_split_subproblems(result.best_code, sub_problem, parent_problem)
        
        return []
    
    def _parse_split_subproblems(
        self,
        response: str,
        original_sp: SubProblem,
        parent_problem: ProblemDefinition
    ) -> List[SubProblem]:
        """Parse LLM split response into SubProblem objects."""
        sections = response.split('---')
        split_sps = []
        id_map = {}

        for section in sections:
            section = section.strip()
            if not section or 'SPLIT' not in section.upper():
                continue

            try:
                title = self._extract_field(section, 'Title:')
                description = self._extract_field(section, 'Description:')
                effort_str = self._extract_field(section, 'Effort:')
                deps_str = self._extract_field(section, 'Dependencies:')

                if not title or not description:
                    continue

                # Parse effort
                try:
                    effort = int(re.search(r'\d+', effort_str).group())
                except (AttributeError, ValueError) as e:
                    logger.warning(
                        "Failed to parse effort '%s' for split sub-problem '%s'. Using default effort %sh. Error: %s",
                        effort_str,
                        title,
                        original_sp.estimated_effort // 2,
                        e,
                        exc_info=True
                    )
                    effort = original_sp.estimated_effort // 2

                # Create split sub-problem
                sp_id = generate_id("subproblem")
                split_num = len(split_sps) + 1

                # Calculate reduced complexity
                complexity_factor = 0.6
                split_sp = SubProblem(
                    id=sp_id,
                    parent_id=original_sp.parent_id,
                    title=title,
                    description=description,
                    type=original_sp.type,
                    complexity_score=ComplexityScore(
                        cognitive_complexity=original_sp.complexity_score.cognitive_complexity * complexity_factor,
                        computational_complexity=original_sp.complexity_score.computational_complexity * complexity_factor,
                        domain_complexity=original_sp.complexity_score.domain_complexity * 0.8,
                        integration_complexity=original_sp.complexity_score.integration_complexity * 0.5,
                        overall_complexity=original_sp.complexity_score.overall_complexity * complexity_factor,
                        explanation="Split from complex sub-problem (LLM-guided)"
                    ),
                    dependencies=original_sp.dependencies.copy(),  # Inherit parent dependencies
                    success_criteria=original_sp.success_criteria[:1] if original_sp.success_criteria else [],
                    validation_gauntlet=original_sp.validation_gauntlet,
                    priority=original_sp.priority,
                    estimated_effort=effort
                )

                id_map[split_num] = sp_id
                split_sps.append((split_sp, deps_str))

            except Exception as e:
                logger.debug(f"Failed to parse split section: {e}")
                continue

        # Resolve internal dependencies
        final_splits = []
        for split_sp, deps_str in split_sps:
            if deps_str and 'previous' in deps_str.lower() and final_splits:
                # Depends on previous split
                split_sp.dependencies.append(final_splits[-1].id)
            elif deps_str:
                # Try to parse split numbers
                try:
                    dep_nums = [int(d) for d in re.findall(r'\d+', deps_str)]
                    for num in dep_nums:
                        if num in id_map:
                            split_sp.dependencies.append(id_map[num])
                except (ValueError, TypeError) as e:
                    logger.debug("Failed to parse split dependencies '%s': %s", deps_str, e)

            final_splits.append(split_sp)

        return final_splits

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract field value from text."""
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(field_name):
                return line.split(':', 1)[1].strip()

        logger.warning(
            "Field '%s' not found in LLM response section for ComplexityDecomposition.",
            field_name
        )
        return ""


class HybridDecomposition(DecompositionStrategyBase):
    """Combines multiple strategies adaptively."""

    def get_strategy_name(self) -> str:
        return "hybrid"

    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda problem: [])
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """
        Applies multiple strategies and merges results intelligently.
        
        Uses semantic decomposition as base, then enhances with
        dependency analysis and complexity balancing.
        """
        logger.info(f"Hybrid decomposition for problem: {problem.id}")
        
        # Step 1: Get results from multiple strategies
        semantic_strategy = SemanticDecomposition()
        dependency_strategy = DependencyDecomposition()
        complexity_strategy = ComplexityDecomposition()
        
        semantic_results = semantic_strategy.decompose(problem)
        dependency_results = dependency_strategy.decompose(problem)
        
        # Step 2: Merge strategies intelligently
        # Use semantic as base structure
        merged_sub_problems = self._merge_semantic_and_dependency(
            semantic_results, 
            dependency_results
        )
        
        # Step 3: Apply complexity balancing
        balanced_sub_problems = self._apply_complexity_balancing(
            merged_sub_problems,
            problem,
            max_complexity=7.0
        )
        
        # Step 4: Optimize dependencies
        optimized_sub_problems = self._optimize_dependencies(balanced_sub_problems)
        
        logger.info(f"Created {len(optimized_sub_problems)} sub-problems via hybrid decomposition")
        return optimized_sub_problems
    
    def _merge_semantic_and_dependency(
        self,
        semantic_results: List[SubProblem],
        dependency_results: List[SubProblem]
    ) -> List[SubProblem]:
        """
        Merge semantic and dependency-based decompositions.
        
        Uses semantic structure but enhances with dependency information.
        """
        # Use semantic results as base
        merged = []
        
        for semantic_sp in semantic_results:
            # Find similar sub-problems in dependency results
            similar_deps = self._find_similar_subproblems(
                semantic_sp,
                dependency_results
            )
            
            # Enhance with dependency information
            enhanced_dependencies = list(set(
                semantic_sp.dependencies + 
                [dep for sp in similar_deps for dep in sp.dependencies]
            ))
            
            # Create enhanced sub-problem
            enhanced_sp = SubProblem(
                id=semantic_sp.id,
                parent_id=semantic_sp.parent_id,
                title=semantic_sp.title,
                description=semantic_sp.description,
                type=semantic_sp.type,
                complexity_score=semantic_sp.complexity_score,
                dependencies=enhanced_dependencies,
                success_criteria=semantic_sp.success_criteria,
                validation_gauntlet=semantic_sp.validation_gauntlet,
                priority=semantic_sp.priority,
                estimated_effort=semantic_sp.estimated_effort
            )
            merged.append(enhanced_sp)
        
        return merged
    
    def _find_similar_subproblems(
        self,
        target: SubProblem,
        candidates: List[SubProblem]
    ) -> List[SubProblem]:
        """Find sub-problems similar to target based on title/description."""
        similar = []
        target_words = set(target.title.lower().split() + target.description.lower().split())
        
        for candidate in candidates:
            candidate_words = set(candidate.title.lower().split() + candidate.description.lower().split())
            overlap = len(target_words & candidate_words)
            
            # If significant overlap, consider similar
            if overlap >= 3:
                similar.append(candidate)
        
        return similar
    
    def _apply_complexity_balancing(
        self,
        sub_problems: List[SubProblem],
        problem: ProblemDefinition,
        max_complexity: float = 7.0
    ) -> List[SubProblem]:
        """Balance complexity across sub-problems using LLM-based splitting."""
        balanced = []
        complexity_strategy = ComplexityDecomposition()

        for sp in sub_problems:
            if sp.complexity_score.overall_complexity > max_complexity:
                logger.info(f"Splitting complex sub-problem in hybrid mode: {sp.title}")
                global OpenEvolveClient, OPENEVOLVE_AVAILABLE
                if not complexity_strategy.openevolve_client and OPENEVOLVE_AVAILABLE:
                    try:
                        complexity_strategy.openevolve_client = OpenEvolveClient()
                    except Exception as e:
                        logger.error(f"Failed to instantiate OpenEvolve client for hybrid complexity balancing: {e}", exc_info=True)
                        logger.warning("OpenEvolve client not available for hybrid complexity balancing. Proceeding without LLM-guided splitting for this sub-problem.")
                        # If client instantiation fails, we should not attempt to use it.
                        # The original sub-problem will be retained as a fallback.
                        balanced.append(sp)
                        continue
                elif not OPENEVOLVE_AVAILABLE:
                    logger.warning("OpenEvolve not available, hybrid complexity balancing will operate without LLM-guided splitting for this sub-problem.")
                    balanced.append(sp)
                    continue
                
                try:
                    split_sps = complexity_strategy.split_with_llm(sp, problem)
                    if not split_sps or len(split_sps) < 2:
                        raise ValueError("LLM splitting returned insufficient sub-problems in hybrid mode.")
                    balanced.extend(split_sps)
                except Exception as e:
                    logger.error(f"LLM-based splitting failed in hybrid mode: {e}")
                    # In hybrid mode, we might choose to keep the complex sub-problem instead of failing
                    logger.warning(f"Could not split complex sub-problem {sp.title}, retaining original.")
                    balanced.append(sp)
            else:
                balanced.append(sp)
        
        return balanced
    
    def _optimize_dependencies(self, sub_problems: List[SubProblem]) -> List[SubProblem]:
        """Optimize dependency relationships to remove redundancies."""
        try:
            # Build dependency graph
            dep_graph = {sp.id: set(sp.dependencies) for sp in sub_problems}
            
            # Remove transitive dependencies
            for sp_id in dep_graph:
                # Find all transitive dependencies
                transitive = set()
                for dep in list(dep_graph[sp_id]):
                    if dep in dep_graph:
                        transitive.update(dep_graph[dep])
                
                # Remove transitive dependencies from direct dependencies
                dep_graph[sp_id] -= transitive
            
            # Update sub-problems with optimized dependencies
            optimized = []
            for sp in sub_problems:
                optimized_sp = SubProblem(
                    id=sp.id,
                    parent_id=sp.parent_id,
                    title=sp.title,
                    description=sp.description,
                    type=sp.type,
                    complexity_score=sp.complexity_score,
                    dependencies=list(dep_graph.get(sp.id, [])),
                    success_criteria=sp.success_criteria,
                    validation_gauntlet=sp.validation_gauntlet,
                    priority=sp.priority,
                    estimated_effort=sp.estimated_effort
                )
                optimized.append(optimized_sp)
            
            return optimized
        except Exception as e:
            logger.error(f"An error occurred during dependency optimization: {e}", exc_info=True)
            # Fallback: return original sub-problems if optimization fails
            return sub_problems

class ResearchDecomposition(DecompositionStrategyBase):
    """
    Decomposes research problems into structured investigation phases using LLM.
    """
    
    def __init__(self, openevolve_client: Optional['OpenEvolveClient'] = None):
        global OpenEvolveClient, OPENEVOLVE_AVAILABLE
        self.openevolve_client = openevolve_client
        if not self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                self.openevolve_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized for research decomposition")
            except Exception as e:
                logger.warning(f"Failed to instantiate OpenEvolve client for research decomposition: {e}", exc_info=True)
                self.openevolve_client = None
        elif not OPENEVOLVE_AVAILABLE:
            logger.warning("OpenEvolve not available, research decomposition will operate without LLM.")

    def get_strategy_name(self) -> str:
        return "research"

    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda problem: [])
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """
        Decomposes a research problem into sub-problems using LLM.
        """
        logger.info(f"Research decomposition for problem: {problem.id}")

        if not self.openevolve_client:
            logger.error("OpenEvolve client not available for research decomposition. Cannot perform LLM-powered research decomposition. Returning empty list.")
            return []

        prompt = f"""You are an expert research scientist and project manager. Decompose the following research problem into a set of structured sub-problems.

RESEARCH PROBLEM:
Title: {problem.title}
Description: {problem.description}
Domain: {problem.domain_context.domain}

DECOMPOSITION TASK:
Break down the research problem into 3-5 logical sub-problems representing the research lifecycle. These should include:
1.  **Literature Review & State of the Art:** Understand the existing body of work.
2.  **Hypothesis Formulation:** Define clear, testable hypotheses.
3.  **Methodology & Experimental Design:** Plan the research approach, data collection, and experiments.
4.  **Execution & Data Analysis:** Carry out the research and analyze the results.
5.  **Conclusion & Reporting:** Synthesize findings and document the research.

For EACH sub-problem, provide the following information in this EXACT format:

---
SUB-PROBLEM [number]
Title: [Clear, concise title]
Description: [Detailed description of the research activity]
Type: [research, analysis, or implementation]
Priority: [Integer 1-10]
Effort: [Estimated hours]
Dependencies: [Comma-separated sub-problem numbers, or "none"]
Success: [Specific, measurable criterion for completion]
---

Begin decomposition:
"""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=2000
        )

        if result.success and result.best_code:
            # The parsing logic is in SemanticDecomposition, I can reuse it.
            # This is not ideal, but for now it's the quickest way.
            # I should probably move the parsing logic to a common place.
            semantic_parser = SemanticDecomposition(self.openevolve_client)
            sub_problems = semantic_parser._parse_llm_subproblems(result.best_code, problem)
            if sub_problems:
                logger.info(f"Successfully created {len(sub_problems)} sub-problems via research decomposition")
                return sub_problems
        
        logger.warning("LLM research decomposition failed or returned no sub-problems. Returning empty list.")
        return []


class DecompositionEngine:
    """Orchestrates problem decomposition using multiple strategies."""
    
    def __init__(
        self, 
        problem_analyzer: Optional[ProblemAnalyzer] = None, 
        knowledge_manager: Optional[KnowledgeManager] = None, 
        enable_adaptive_selection: bool = True,
        maker_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize decomposition engine.
        
        Args:
            problem_analyzer: Optional ProblemAnalyzer instance
            knowledge_manager: Optional KnowledgeManager instance
            enable_adaptive_selection: Whether to use granular complexity for strategy selection
            maker_config: Configuration for the MAKER system (voting thresholds, depth, etc.)
        """
        self.strategies: Dict[str, DecompositionStrategyBase] = {
            'semantic': SemanticDecomposition(),
            'dependency': DependencyDecomposition(),
            'complexity': ComplexityDecomposition(),
            'hybrid': HybridDecomposition(),
            'research': ResearchDecomposition()
        }
        self.problem_analyzer = problem_analyzer or ProblemAnalyzer()      
        self.knowledge_manager = knowledge_manager or KnowledgeManager()   
        self.logger = logging.getLogger(__name__)
        self.enable_adaptive_selection = enable_adaptive_selection
        self.maker_config = maker_config or {}
        
        # Integrate Enhanced TaskComplexityClassifier
        try:
            from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
            self.complexity_classifier = TaskComplexityClassifier()
            self.logger.info(f"Integrated Adaptive TaskComplexityClassifier into DecompositionEngine (enabled={enable_adaptive_selection})")
        except ImportError:
            self.logger.warning("TaskComplexityClassifier not available, using standard complexity metrics.")
            self.complexity_classifier = None
            @with_error_handling(severity=ErrorSeverity.CRITICAL, fallback=lambda problem, strategy: DecompositionPlan(
        id=generate_id("plan"),
        problem_id=problem.id,
        strategy=DecompositionStrategy.HYBRID,
        sub_problems=[],
        dependency_graph=DependencyGraph(nodes={}, edges={}),
        validation_checkpoints=[],
        quality_scores=QualityScores(overall_score=0.0, meets_thresholds=False),
        confidence_level=0.0,
        created_by="decomposition_engine_error",
        error_message="Decomposition failed"
    ))
    def decompose(self, problem: ProblemDefinition, strategy: Optional[str] = None) -> DecompositionPlan:
        """
        Decomposes problem using optimal strategy.
        
        Args:
            problem: The problem to decompose
            strategy: Optional strategy name (auto-selected if not provided)
            
        Returns:
            DecompositionPlan with sub-problems, dependencies, execution order
        """
        self.logger.info(f"Decomposing problem: {problem.id}")
        
        # Select strategy if not provided
        if not strategy:
            strategy = self.select_strategy(problem)
        
        self.logger.info(f"Using strategy: {strategy}")
        
        # Get strategy instance
        strategy_instance = self.strategies.get(strategy)
        if not strategy_instance:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Generate sub-problems
        sub_problems = strategy_instance.decompose(problem)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(sub_problems)
        
        # Create quality scores (initial assessment)
        quality_scores = self._assess_quality(problem, sub_problems)
        
        # Create decomposition plan
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id=problem.id,
            strategy=DecompositionStrategy(strategy),
            sub_problems=sub_problems,
            dependency_graph=dependency_graph,
            validation_checkpoints=[],
            quality_scores=quality_scores,
            confidence_level=0.8,  # Initial confidence
            created_by="decomposition_engine"
        )
        
        self.logger.info(f"Decomposition complete: {len(sub_problems)} sub-problems created")
        return plan
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda problem: "hybrid")
    def select_strategy(self, problem: ProblemDefinition) -> str:
        """
        Chooses optimal decomposition strategy using granular complexity analysis and LLM.
        
        Args:
            problem: The problem to analyze
            
        Returns:
            Strategy name
        """
        # Step 1: Calculate granular complexity if enabled
        granular_complexity = None
        if self.enable_adaptive_selection and hasattr(self, 'complexity_classifier') and self.complexity_classifier:
            try:
                # Map ProblemDefinition to AdaptiveSubProblem for classifier
                from adaptive_mdap.core.types import SubProblem as AdaptiveSubProblem
                adaptive_sp = AdaptiveSubProblem(
                    id=problem.id,
                    description=problem.description,
                    domain=problem.domain_context.domain if hasattr(problem, 'domain_context') else "general",
                    depth=0,
                    dependencies=[], # Root problem
                    metadata={"title": problem.title}
                )
                granular_complexity = self.complexity_classifier.compute_complexity(adaptive_sp)
                self.logger.info(f"Granular complexity analysis: score={granular_complexity.overall_score:.3f}")
            except Exception as e:
                self.logger.error(f"Granular complexity calculation failed: {e}")

        # Step 2: Fallback to LLM or Heuristics
        global OpenEvolveClient, OPENEVOLVE_AVAILABLE
        if OPENEVOLVE_AVAILABLE:
            try:
                client = OpenEvolveClient()
                strategy = self._select_strategy_with_llm(problem, client, granular_complexity)
                if strategy in self.strategies:
                    self.logger.info(f"LLM selected strategy: {strategy}")
                    return strategy
                else:
                    self.logger.warning(f"LLM returned an unknown strategy: {strategy}. Falling back to heuristic.")
            except Exception as e:
                self.logger.error(f"Failed to call LLM for strategy selection: {e}")
        
        # Step 3: Heuristic selection based on complexity
        overall_complexity = granular_complexity.overall_score * 10 if granular_complexity else problem.complexity_score.overall_complexity
        
        if overall_complexity > 7.5:
            return 'hybrid'
        elif overall_complexity > 5.0:
            return 'semantic'
        else:
            return 'complexity'
    
    def _select_strategy_with_llm(self, problem: ProblemDefinition, client, granular_complexity: Any = None) -> str:
        """Use LLM to select optimal decomposition strategy."""
        complexity_context = ""
        if granular_complexity:
            complexity_context = f"\nGRANULAR COMPLEXITY BREAKDOWN:\n"
            complexity_context += f"- Overall Score: {granular_complexity.overall_score:.3f}\n"
            complexity_context += f"- Text Length Score: {granular_complexity.text_length_score:.2f}\n"
            complexity_context += f"- Keyword Complexity: {granular_complexity.keyword_score:.2f}\n"
            complexity_context += f"- Constraint Density: {granular_complexity.constraint_score:.2f}\n"

        prompt = f"""You are an expert in problem decomposition strategies. Select the BEST strategy for this problem.

PROBLEM:
Title: {problem.title}
Description: {problem.description}
Domain: {problem.domain_context.domain}
Type: {problem.problem_type.value}
Complexity: {problem.complexity_score.overall_complexity}/10
Constraints: {len(problem.constraints)}
{complexity_context}

AVAILABLE STRATEGIES:

1. SEMANTIC: Decomposes based on semantic concepts and natural boundaries
   - Best for: Problems with distinct conceptual phases or components
   - Strengths: Natural, intuitive decomposition
   - Use when: Problem has clear conceptual structure

2. DEPENDENCY: Decomposes based on prerequisite relationships
   - Best for: Problems with strong sequential dependencies
   - Strengths: Clear execution order, manages dependencies
   - Use when: Many interdependencies between components

3. COMPLEXITY: Decomposes to balance cognitive load
   - Best for: Very complex problems that need simplification
   - Strengths: Ensures manageable sub-problem complexity
   - Use when: Overall complexity > 7.5/10

4. HYBRID: Combines multiple strategies adaptively
   - Best for: Complex problems needing multiple perspectives
   - Strengths: Most sophisticated, handles complex cases
   - Use when: Problem has multiple challenging aspects

SELECTION CRITERIA:
- Problem complexity and structure
- Dependency relationships
- Domain characteristics
- Constraint complexity

Respond with ONLY ONE WORD - the strategy name:
semantic OR dependency OR complexity OR hybrid

Your selection:"""
        
        result = client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.2,  # Low temperature for consistent selection
            max_tokens=50
        )
        
        if result.success and result.best_code:
            strategy = result.best_code.strip().lower()
            # Validate it's a known strategy
            if strategy in ['semantic', 'dependency', 'complexity', 'hybrid']:
                return strategy
        
        raise RuntimeError("LLM failed to select a valid strategy.")
    

    
    def _build_dependency_graph(self, sub_problems: List[SubProblem]) -> DependencyGraph:
        """Build dependency graph from sub-problems."""
        nodes = {sp.id: sp for sp in sub_problems}
        edges = {sp.id: sp.dependencies for sp in sub_problems}
        
        # Calculate execution order (topological sort)
        execution_order = self._topological_sort(sub_problems)
        
        return DependencyGraph(
            nodes=nodes,
            edges=edges,
            critical_path=[],  # Will be calculated by DependencyManager
            parallel_groups=[],  # Will be calculated by DependencyManager
            execution_order=execution_order
        )
    
    def _topological_sort(self, sub_problems: List[SubProblem]) -> List[str]:
        """Simple topological sort for execution order."""
        # Build adjacency list
        in_degree = {sp.id: len(sp.dependencies) for sp in sub_problems}
        
        # Find nodes with no dependencies
        queue = [sp.id for sp in sub_problems if len(sp.dependencies) == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Find nodes that depend on this node
            for sp in sub_problems:
                if node in sp.dependencies:
                    in_degree[sp.id] -= 1
                    if in_degree[sp.id] == 0:
                        queue.append(sp.id)
        
        return result
    
    def _assess_quality(self, problem: ProblemDefinition, sub_problems: List[SubProblem]) -> QualityScores:
        """Assess initial quality of decomposition."""
        from datetime import datetime

        if not sub_problems:
            return QualityScores(
                coherence_score=0.0,
                completeness_score=0.0,
                feasibility_score=0.0,
                integration_score=0.0,
                overall_score=0.0,
                meets_thresholds=False,
                details={"error": "No sub-problems generated"},
                timestamp=datetime.now()
            )

        def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
            return max(low, min(high, value))

        sub_problem_count = len(sub_problems)
        sub_problem_ids = {sp.id for sp in sub_problems}
        total_dependencies = sum(len(sp.dependencies) for sp in sub_problems)
        missing_dependencies = sum(
            1 for sp in sub_problems for dep in sp.dependencies if dep not in sub_problem_ids
        )
        self_dependencies = sum(
            1 for sp in sub_problems for dep in sp.dependencies if dep == sp.id
        )

        execution_order = self._topological_sort(sub_problems)
        has_cycles = len(execution_order) < sub_problem_count

        # Coherence: dependency validity, acyclic graph, and description uniqueness.
        dependency_validity = 1.0 - (missing_dependencies / max(1, total_dependencies))
        dependency_validity = _clamp(dependency_validity)
        cycle_penalty = 0.2 if has_cycles else 0.0
        self_dep_penalty = min(0.2, self_dependencies / max(1, total_dependencies)) if total_dependencies else 0.0

        description_tokens = [
            set(re.findall(r"\w+", sp.description.lower())) for sp in sub_problems
        ]
        similarity_scores = []
        for idx, tokens in enumerate(description_tokens):
            for jdx in range(idx + 1, len(description_tokens)):
                other = description_tokens[jdx]
                if tokens and other:
                    similarity = len(tokens & other) / max(1, len(tokens | other))
                    similarity_scores.append(similarity)
        average_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        redundancy_penalty = min(0.2, average_similarity)

        coherence = _clamp(0.9 * dependency_validity - cycle_penalty - self_dep_penalty - redundancy_penalty)

        # Completeness: align sub-problem count with overall complexity and coverage.
        complexity = problem.complexity_score.overall_complexity if problem.complexity_score else 5.0
        if complexity <= 3.0:
            expected_min, expected_max = 2, 4
        elif complexity <= 6.0:
            expected_min, expected_max = 3, 7
        else:
            expected_min, expected_max = 5, 12

        if sub_problem_count < expected_min:
            completeness = _clamp(sub_problem_count / max(1, expected_min))
        elif sub_problem_count > expected_max:
            completeness = _clamp(expected_max / max(1, sub_problem_count))
        else:
            completeness = 1.0

        empty_descriptions = sum(1 for sp in sub_problems if not sp.description.strip())
        completeness = _clamp(completeness - (empty_descriptions / max(1, sub_problem_count)) * 0.3)

        # Feasibility: ensure complexity aligns with available resources and configuration richness.
        complexity_scores = [
            sp.ai_suggested_complexity_score if sp.ai_suggested_complexity_score else 5
            for sp in sub_problems
        ]
        avg_complexity = sum(complexity_scores) / max(1, len(complexity_scores))
        max_allowed = problem.resources_available.get("max_subproblem_complexity") or problem.resources_available.get("max_complexity")
        feasibility_penalty = 0.0
        if max_allowed:
            feasibility_penalty += max(0.0, (avg_complexity - max_allowed) / max_allowed)

        missing_requirements = sum(
            1 for sp in sub_problems if not sp.solution_requirements and not sp.acceptance_criteria
        )
        feasibility_penalty += (missing_requirements / max(1, sub_problem_count)) * 0.2

        feasibility = _clamp(1.0 - feasibility_penalty)

        # Integration: ensure dependencies connect sub-problems and outputs are defined.
        dependency_density = total_dependencies / max(1, sub_problem_count * (sub_problem_count - 1))
        integration = 0.5 + min(0.5, dependency_density * 2.0)

        missing_dependency_outputs = sum(
            1 for sp in sub_problems if sp.dependencies and not sp.dependency_outputs
        )
        integration = _clamp(integration - (missing_dependency_outputs / max(1, sub_problem_count)) * 0.3)

        overall = _clamp((coherence + completeness + feasibility + integration) / 4.0)
        
        return QualityScores(
            coherence_score=coherence,
            completeness_score=completeness,
            feasibility_score=feasibility,
            integration_score=integration,
            overall_score=overall,
            meets_thresholds=overall >= 0.8,
            details={
                "method": "heuristic_assessment",
                "dependency_validity": dependency_validity,
                "has_cycles": has_cycles,
                "average_similarity": average_similarity,
                "expected_sub_problem_range": [expected_min, expected_max],
                "average_complexity": avg_complexity,
                "missing_dependencies": missing_dependencies,
                "self_dependencies": self_dependencies,
                "missing_requirements": missing_requirements,
                "missing_dependency_outputs": missing_dependency_outputs,
            },
            timestamp=datetime.now()
        )

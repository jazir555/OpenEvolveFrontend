"""
Workflow Knowledge Extractor - Stage 6 Knowledge Extraction

This module extracts knowledge artifacts from workflow executions across all stages.
It uses LLM-based semantic extraction to identify patterns, team performance insights,
and gauntlet effectiveness metrics.

Enhanced with OneKE integration for schema-guided domain knowledge extraction.
"""

import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio
import dataclasses

# Import from workflow_structures - handle missing optional dependencies gracefully
try:
    from workflow_structures import (
        WorkflowState,
        SolutionPatternArtifact,
        TeamPerformanceArtifact,
        GauntletEffectivenessArtifact,
        CritiqueInsightArtifact,
        KnowledgeArtifact,
        Team,
        GauntletDefinition,
        SolutionAttempt,
        CritiqueReport,
        VerificationReport,
        DecompositionPlan,

    )
except ImportError as e:
    logging.warning(f"Could not import from workflow_structures: {e}")
    # Define minimal classes for type hints if imports fail
    WorkflowState = Any
    SolutionPatternArtifact = Any
    TeamPerformanceArtifact = Any
    GauntletEffectivenessArtifact = Any
    CritiqueInsightArtifact = Any
    KnowledgeArtifact = Any
    Team = Any
    GauntletDefinition = Any
    SolutionAttempt = Any
    CritiqueReport = Any
    VerificationReport = Any
    DecompositionPlan = Any
    ProblemAnalysis = Any

# Optional ACE client import
try:
    from ace_client import ACEClient
except ImportError:
    ACEClient = Any
    logging.warning("ACE client not available. Install ace_client for enhanced extraction.")

# Optional DSPy import
try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    DSPY_AVAILABLE = True
except ImportError:
    dspy = None
    BootstrapFewShot = None
    DSPY_AVAILABLE = False
    logging.warning("DSPy not available. Install dspy for enhanced programmatic prompting.")

# Optional knowledge engine import
try:
    from knowledge_engine.orchestration import KnowledgeEngine
except ImportError:
    KnowledgeEngine = Any
    logging.warning("KnowledgeEngine not available. Install knowledge_engine for enhanced extraction.")


logger = logging.getLogger(__name__)


class KnowledgeArtifactManager:
    """
    Manager for storing and retrieving knowledge artifacts.
    
    This class provides a simple in-memory and file-based storage mechanism
    for knowledge artifacts extracted from workflows.
    
    Attributes:
        db_path: Path to the artifact database
        artifacts: In-memory cache of artifacts
    """
    
    def __init__(self, db_path: str = "./knowledge_artifacts.db"):
        """
        Initialize the artifact manager.
        
        Args:
            db_path: Path to the artifact database
        """
        self.db_path = db_path
        self.artifacts: Dict[str, KnowledgeArtifact] = {}
        self.solution_patterns: Dict[str, SolutionPatternArtifact] = {}
        self.team_performances: Dict[str, TeamPerformanceArtifact] = {}
        self.gauntlet_effectiveness: Dict[str, GauntletEffectivenessArtifact] = {}
    
    def create_solution_pattern(self, artifact: SolutionPatternArtifact) -> str:
        """Store a solution pattern artifact."""
        artifact_id = getattr(artifact, 'artifact_id', str(uuid.uuid4()))
        self.solution_patterns[artifact_id] = artifact
        return artifact_id
    
    def create_team_performance(self, artifact: TeamPerformanceArtifact) -> str:
        """Store a team performance artifact."""
        artifact_id = getattr(artifact, 'artifact_id', str(uuid.uuid4()))
        self.team_performances[artifact_id] = artifact
        return artifact_id
    
    def create_gauntlet_effectiveness(self, artifact: GauntletEffectivenessArtifact) -> str:
        """Store a gauntlet effectiveness artifact."""
        artifact_id = getattr(artifact, 'artifact_id', str(uuid.uuid4()))
        self.gauntlet_effectiveness[artifact_id] = artifact
        return artifact_id
    
    def get_artifact(self, artifact_id: str) -> Optional[KnowledgeArtifact]:
        """Retrieve an artifact by ID."""
        return self.artifacts.get(artifact_id)
    
    def search_artifacts(self, artifact_type: Optional[str] = None,
                        domain: Optional[str] = None) -> List[KnowledgeArtifact]:
        """Search artifacts by type and/or domain."""
        results = []
        for artifact in self.artifacts.values():
            if artifact_type and getattr(artifact, 'artifact_type', None) != artifact_type:
                continue
            if domain and getattr(artifact, 'domain', None) != domain:
                continue
            results.append(artifact)
        return results

    def list_solution_patterns(self, limit: int = 100) -> List[SolutionPatternArtifact]:
        """List solution pattern artifacts."""
        patterns = list(self.solution_patterns.values())
        return patterns[:limit]

    def list_team_performance(self, limit: int = 100) -> List[TeamPerformanceArtifact]:
        """List team performance artifacts."""
        performances = list(self.team_performances.values())
        return performances[:limit]

    def list_gauntlet_effectiveness(self, limit: int = 100) -> List[GauntletEffectivenessArtifact]:
        """List gauntlet effectiveness artifacts."""
        effectiveness = list(self.gauntlet_effectiveness.values())
        return effectiveness[:limit]


class WorkflowKnowledgeExtractor:
    """
    Extracts knowledge artifacts from workflow executions.
    
    This class analyzes workflow data from all stages to extract:
    - Solution patterns (successful problem-solving approaches)
    - Team performance insights (which teams work best for which problems)
    - Gauntlet effectiveness (which quality checks work best)
    - Domain detection patterns
    - Decomposition strategies
    - Self-healing patterns
    - Learning patterns
    
    Attributes:
        knowledge_engine: Optional knowledge engine for enhanced extraction
        ace_client: Optional ACE client for LLM-based extraction
        artifact_manager: Manager for storing/retrieving artifacts
        extraction_prompts: Dictionary of prompts for each extraction type
    """
    
    def __init__(self, knowledge_engine: Optional[Any] = None, ace_client: Optional[Any] = None):
        """
        Initialize the knowledge extractor.
        
        Args:
            knowledge_engine: Optional knowledge engine for enhanced extraction
            ace_client: Optional ACE client for LLM-based extraction
        """
        self.knowledge_engine = knowledge_engine
        self.ace_client = ace_client
        self.artifact_manager = KnowledgeArtifactManager()
        self.extraction_prompts = self._init_extraction_prompts()
        self._oneke_initialized = False
        self.oneke_bridge = None
        
        # Try to initialize OneKE if available
        try:
            from integrations.oneke import OneKEBridge
            self.oneke_bridge = OneKEBridge()
            logger.info("OneKE bridge created. Call ensure_oneke_initialized() before use.")
        except ImportError:
            logger.debug("OneKE not available. Install with: pip install integrations/oneke")
    
    def _init_extraction_prompts(self) -> Dict[str, str]:
        """Initialize LLM prompts for different extraction tasks."""
        return {
            "solution_pattern": """
Analyze the following workflow execution and extract solution patterns:

Problem: {problem_statement}
Decomposition Strategy: {decomposition_strategy}
Final Solution: {final_solution}
Success: {success}

Extract and describe:
1. Problem characteristics (domain, complexity, constraints)
2. Solution approach (high-level strategy)
3. Decomposition strategy (ROMA/MAKER/MDAP)
4. Key code patterns
5. Optimization techniques used
6. Typical refinements needed

Return as JSON with keys: problem_characteristics, solution_approach, decomposition_strategy,
code_patterns, optimization_techniques, typical_refinements, domain, complexity (1-10).
""",
            "decomposition_strategy": """
Analyze the following decomposition strategy and extract insights:

Problem: {problem_statement}
Strategy: {decomposition_strategy}
Framework: {framework} (ROMA/MAKER/MDAP)
Sub-problems: {num_sub_problems}
Success: {success}

Extract and describe:
1. Decomposition framework used (ROMA/MAKER/MDAP)
2. Decision points and rationale
3. Domain-specific approaches
4. Sub-problem dependencies
5. Integration strategy
6. Effectiveness metrics

Return as JSON with keys: framework, decision_points, rationale, domain_approaches,
dependencies, integration_strategy, effectiveness, complexity (1-10).
""",
            "team_performance": """
Analyze the following team's performance:

Team: {team_id}
Composition: {team_composition}
Problems Solved: {num_solved}
Total Problems: {total_problems}
Quality Metrics: {quality_metrics}
Domains: {domains}
Complexities: {complexities}

Extract:
1. Team velocity (solutions/time)
2. Quality metrics (success rate, error rate)
3. Optimal domains
4. Skill gaps
5. Training recommendations

Return as JSON with keys: velocity, quality_metrics, optimal_domains, skill_gaps, training_recommendations.
""",
            "gauntlet_effectiveness": """
Analyze the following gauntlet's effectiveness:

Gauntlet: {gauntlet_id}
Type: {gauntlet_type}
Total Checks: {total_checks}
Issues Caught: {issues_caught}
False Positives: {false_positives}
Execution Time: {execution_time}

Extract:
1. Catch rate (issues caught / total issues)
2. False positive rate
3. Problem type effectiveness
4. Rule effectiveness (if available)
5. Recommended improvements

Return as JSON with keys: catch_rate, false_positive_rate, problem_type_effectiveness,
rule_effectiveness, recommended_improvements.
""",
            "domain_detection": """
Analyze the following problem statement and extract domain detection patterns:

Problem: {problem_statement}
Detected Domain: {detected_domain}
Confidence: {confidence}

Extract:
1. Domain keywords and indicators
2. Problem type classification
3. Complexity indicators
4. Domain-specific requirements
5. Related domains

Return as JSON with keys: domain_keywords, problem_type, complexity_indicators, 
domain_requirements, related_domains.
""",
            "self_healing": """
Analyze the following self-healing execution and extract patterns:

Problem: {problem_statement}
Refinement Loop Count: {refinement_count}
Issues Found: {issues_found}
Resolution Strategy: {resolution_strategy}
Success: {success}

Extract:
1. Common issue patterns
2. Effective resolution strategies
3. Refinement loop effectiveness
4. Self-healing triggers
5. Prevention recommendations

Return as JSON with keys: issue_patterns, resolution_strategies, refinement_effectiveness,
healing_triggers, prevention_recommendations.
""",
            "learning_patterns": """
Analyze the following workflow execution and extract learning patterns:

Workflow ID: {workflow_id}
Problem: {problem_statement}
Success: {success}
Execution Time: {execution_time}
Quality Score: {quality_score}

Extract:
1. What worked well
2. What could be improved
3. Key learnings
4. Reusable patterns
5. Adaptation recommendations

Return as JSON with keys: success_factors, improvement_areas, key_learnings,
reusable_patterns, adaptation_recommendations.
""",
        }
    
    # ========== Main Extraction Method ==========
    
    def extract_from_workflow(self, workflow_state: WorkflowState) -> List[KnowledgeArtifact]:
        """
        Extract all knowledge artifacts from a workflow state.
        
        This is the main entry point that extracts artifacts from all stages
        of the workflow where relevant data exists.
        
        Args:
            workflow_state: The workflow state to extract from
            
        Returns:
            List of extracted KnowledgeArtifact instances
        """
        artifacts = []
        
        # Extract from Stage 0: Problem Definition
        try:
            stage_0_artifacts = self._extract_from_stage_0(workflow_state)
            artifacts.extend(stage_0_artifacts)
            logger.debug(f"Extracted {len(stage_0_artifacts)} artifacts from Stage 0")
        except Exception as e:
            logger.warning(f"Failed to extract from Stage 0: {e}")
        
        # Extract from Stage 1: Decomposition
        try:
            stage_1_artifacts = self._extract_from_stage_1(workflow_state)
            artifacts.extend(stage_1_artifacts)
            logger.debug(f"Extracted {len(stage_1_artifacts)} artifacts from Stage 1")
        except Exception as e:
            logger.warning(f"Failed to extract from Stage 1: {e}")
        
        # Skip Stage 2 (Planning) - usually doesn't have extractable patterns
        
        # Extract from Stage 3: Solution & Critique
        try:
            stage_3_artifacts = self._extract_from_stage_3(workflow_state)
            artifacts.extend(stage_3_artifacts)
            logger.debug(f"Extracted {len(stage_3_artifacts)} artifacts from Stage 3")
        except Exception as e:
            logger.warning(f"Failed to extract from Stage 3: {e}")
        
        # Skip Stage 4 (Verification) - patterns extracted in Stage 5
        
        # Extract from Stage 5: Quality Assessment & Self-Healing
        try:
            stage_5_artifacts = self._extract_from_stage_5(workflow_state)
            artifacts.extend(stage_5_artifacts)
            logger.debug(f"Extracted {len(stage_5_artifacts)} artifacts from Stage 5")
        except Exception as e:
            logger.warning(f"Failed to extract from Stage 5: {e}")
        
        # Extract from Stage 6: Execution Results & Learning
        try:
            stage_6_artifacts = self._extract_from_stage_6(workflow_state)
            artifacts.extend(stage_6_artifacts)
            logger.debug(f"Extracted {len(stage_6_artifacts)} artifacts from Stage 6")
        except Exception as e:
            logger.warning(f"Failed to extract from Stage 6: {e}")
        
        return artifacts
    
    # ========== Stage 0: Problem Definition ==========
    
    def _extract_from_stage_0(self, workflow_state: WorkflowState) -> List[KnowledgeArtifact]:
        """
        Extract domain detection patterns from Stage 0 (Problem Definition).
        
        Args:
            workflow_state: The workflow state
            
        Returns:
            List of KnowledgeArtifacts with domain detection patterns
        """
        artifacts = []
        
        if not hasattr(workflow_state, 'problem_statement') or not workflow_state.problem_statement:
            return artifacts
        
        problem_statement = workflow_state.problem_statement
        
        # Classify domain
        domain = self._classify_domain(problem_statement)
        complexity = self._estimate_complexity(problem_statement)
        constraints = self._extract_constraints(problem_statement)
        
        # Create domain detection artifact
        content = {
            "domain": domain,
            "complexity": complexity,
            "constraints": constraints,
            "domain_keywords": self._extract_domain_keywords(problem_statement, domain),
            "problem_type": self._classify_problem_type(problem_statement),
        }
        
        artifact = KnowledgeArtifact(
            artifact_id=f"domain_{uuid.uuid4().hex[:16]}",
            artifact_type="domain_knowledge",
            source_workflow_id=getattr(workflow_state, 'workflow_id', 'unknown'),
            source_stage=0,
            timestamp=datetime.now(),
            confidence=0.8,
            title=f"Domain Detection: {domain}",
            description=f"Domain detection patterns for {domain} problems",
            content=content,
            tags=["domain_detection", domain, f"complexity_{complexity}"],
        )
        artifacts.append(artifact)
        
        # Use LLM for enhanced extraction if available
        if self.ace_client:
            try:
                prompt = self.extraction_prompts["domain_detection"].format(
                    problem_statement=problem_statement,
                    detected_domain=domain,
                    confidence=0.8
                )
                response = self._call_llm(prompt)
                if response:
                    extracted_data = json.loads(response)
                    content.update({
                        "domain_keywords": extracted_data.get("domain_keywords", []),
                        "problem_type": extracted_data.get("problem_type", "unknown"),
                        "complexity_indicators": extracted_data.get("complexity_indicators", []),
                        "domain_requirements": extracted_data.get("domain_requirements", []),
                        "related_domains": extracted_data.get("related_domains", []),
                    })
                    artifact.content = content
            except Exception as e:
                logger.warning(f"LLM domain detection failed: {e}")
        
        return artifacts
    
    # ========== Stage 1: Decomposition ==========
    
    def _extract_from_stage_1(self, workflow_state: WorkflowState) -> List[KnowledgeArtifact]:
        """
        Extract decomposition strategies from Stage 1 (Decomposition).
        
        Args:
            workflow_state: The workflow state
            
        Returns:
            List of KnowledgeArtifacts with decomposition strategies
        """
        artifacts = []
        
        if not hasattr(workflow_state, 'decomposition_plan') or not workflow_state.decomposition_plan:
            return artifacts
        
        decomposition_plan = workflow_state.decomposition_plan
        
        # Extract decomposition strategy
        strategy = getattr(decomposition_plan, 'decomposition_method', 'unknown')
        framework = getattr(decomposition_plan, 'framework', 'unknown')
        
        # Get sub-problems count
        sub_problems = getattr(decomposition_plan, 'sub_problems', [])
        num_sub_problems = len(sub_problems)
        
        # Create decomposition strategy artifact
        content = {
            "framework": framework,
            "strategy": strategy,
            "num_sub_problems": num_sub_problems,
            "sub_problem_types": [getattr(sp, 'type', 'unknown') for sp in sub_problems],
            "dependencies": getattr(decomposition_plan, 'dependencies', []),
        }
        
        artifact = KnowledgeArtifact(
            artifact_id=f"decomp_{uuid.uuid4().hex[:16]}",
            artifact_type="decomposition_strategy",
            source_workflow_id=getattr(workflow_state, 'workflow_id', 'unknown'),
            source_stage=1,
            timestamp=datetime.now(),
            confidence=0.85,
            title=f"Decomposition Strategy: {framework}",
            description=f"Decomposition strategy using {framework} framework",
            content=content,
            tags=["decomposition", framework, strategy],
        )
        artifacts.append(artifact)
        
        # Use LLM for enhanced extraction if available
        if self.ace_client:
            try:
                prompt = self.extraction_prompts["decomposition_strategy"].format(
                    problem_statement=getattr(workflow_state, 'problem_statement', ''),
                    decomposition_strategy=strategy,
                    framework=framework,
                    num_sub_problems=num_sub_problems,
                    success=getattr(workflow_state, 'status', '') == 'completed'
                )
                response = self._call_llm(prompt)
                if response:
                    extracted_data = json.loads(response)
                    content.update({
                        "decision_points": extracted_data.get("decision_points", []),
                        "rationale": extracted_data.get("rationale", ""),
                        "domain_approaches": extracted_data.get("domain_approaches", []),
                        "integration_strategy": extracted_data.get("integration_strategy", ""),
                        "effectiveness": extracted_data.get("effectiveness", 0.5),
                    })
                    artifact.content = content
            except Exception as e:
                logger.warning(f"LLM decomposition extraction failed: {e}")
        
        return artifacts
    
    # ========== Stage 3: Solution & Critique ==========
    
    def _extract_from_stage_3(self, workflow_state: WorkflowState) -> List[KnowledgeArtifact]:
        """
        Extract solution patterns and critique insights from Stage 3 (Solution & Critique).
        
        Args:
            workflow_state: The workflow state
            
        Returns:
            List of SolutionPatternArtifacts and CritiqueInsightArtifacts
        """
        artifacts = []
        
        # Extract solution patterns from sub-problem solutions
        sub_problem_solutions = getattr(workflow_state, 'sub_problem_solutions', {})
        for sub_problem_id, solution in sub_problem_solutions.items():
            if not solution:
                continue
            
            # Filter for high-quality solutions (>0.8 score)
            quality_score = getattr(solution, 'quality_score', 0.0)
            if quality_score < 0.8:
                continue
            
            # Create solution pattern artifact
            pattern = self._create_solution_pattern_from_solution(
                workflow_state, sub_problem_id, solution
            )
            if pattern:
                artifacts.append(pattern)
        
        # Extract critique insights
        critique_reports = getattr(workflow_state, 'all_critique_reports', [])
        for critique in critique_reports:
            if not critique:
                continue
            
            critique_artifact = self._create_critique_insight(workflow_state, critique)
            if critique_artifact:
                artifacts.append(critique_artifact)
        
        return artifacts
    
    def _create_solution_pattern_from_solution(self, workflow_state: WorkflowState, 
                                                sub_problem_id: str,
                                                solution: SolutionAttempt) -> Optional[SolutionPatternArtifact]:
        """Create a SolutionPatternArtifact from a solution attempt."""
        try:
            content = {
                "sub_problem_id": sub_problem_id,
                "code_language": getattr(solution, 'code_language', 'python'),
                "solution_quality": getattr(solution, 'quality_score', 0.0),
                "approach": getattr(solution, 'approach_description', ''),
            }
            
            # Extract code patterns
            final_code = getattr(solution, 'final_code', '')
            if final_code:
                content["code_patterns"] = self._extract_code_patterns(final_code)
            
            artifact = SolutionPatternArtifact(
                artifact_id=f"pattern_{uuid.uuid4().hex[:16]}",
                artifact_type="solution_pattern",
                source_workflow_id=getattr(workflow_state, 'workflow_id', 'unknown'),
                source_stage=3,
                timestamp=datetime.now(),
                confidence=getattr(solution, 'quality_score', 0.8),
                title=f"Solution Pattern: {sub_problem_id}",
                description=f"Solution pattern from sub-problem {sub_problem_id}",
                content=content,
                pattern_category=getattr(solution, 'approach_type', 'general'),
                problem_domains=[self._classify_domain(getattr(workflow_state, 'problem_statement', ''))],
                approach_signature={"quality_score": getattr(solution, 'quality_score', 0.0)},
                success_rate=1.0 if getattr(solution, 'is_successful', False) else 0.0,
                avg_execution_time=getattr(solution, 'execution_time', 0.0),
                tags=["solution_pattern", sub_problem_id, getattr(solution, 'code_language', 'python')],
            )
            return artifact
        except Exception as e:
            logger.warning(f"Failed to create solution pattern: {e}")
            return None
    
    def _create_critique_insight(self, workflow_state: WorkflowState, 
                                  critique: CritiqueReport) -> Optional[CritiqueInsightArtifact]:
        """Create a CritiqueInsightArtifact from a critique report."""
        try:
            content = {
                "critique_type": getattr(critique, 'critique_type', 'general'),
                "issues_found": getattr(critique, 'issues', []),
                "suggestions": getattr(critique, 'suggestions', []),
                "severity": getattr(critique, 'severity', 'medium'),
            }
            
            artifact = CritiqueInsightArtifact(
                artifact_id=f"critique_{uuid.uuid4().hex[:16]}",
                artifact_type="critique_insight",
                source_workflow_id=getattr(workflow_state, 'workflow_id', 'unknown'),
                source_stage=3,
                timestamp=datetime.now(),
                confidence=0.75,
                title=f"Critique Insight: {getattr(critique, 'critique_type', 'general')}",
                description=f"Insights from {getattr(critique, 'critique_type', 'general')} critique",
                content=content,
                critique_type=getattr(critique, 'critique_type', 'general'),
                common_issues=[issue.get('type', 'unknown') for issue in getattr(critique, 'issues', [])],
                improvement_suggestions=getattr(critique, 'suggestions', []),
                tags=["critique", getattr(critique, 'critique_type', 'general')],
            )
            return artifact
        except Exception as e:
            logger.warning(f"Failed to create critique insight: {e}")
            return None
    
    # ========== Stage 5: Quality Assessment & Self-Healing ==========
    
    def _extract_from_stage_5(self, workflow_state: WorkflowState) -> List[KnowledgeArtifact]:
        """
        Extract self-healing patterns from Stage 5 (Quality Assessment).
        
        Args:
            workflow_state: The workflow state
            
        Returns:
            List of KnowledgeArtifacts with self-healing patterns
        """
        artifacts = []
        
        refinement_count = getattr(workflow_state, 'refinement_loop_count', 0)
        if refinement_count == 0:
            return artifacts  # No self-healing occurred
        
        # Extract self-healing patterns
        content = {
            "refinement_loop_count": refinement_count,
            "max_refinement_loops": getattr(workflow_state, 'max_refinement_loops', 3),
            "refinement_effective": refinement_count < getattr(workflow_state, 'max_refinement_loops', 3),
            "verification_reports_count": len(getattr(workflow_state, 'all_verification_reports', [])),
        }
        
        # Analyze issues found
        issues_found = []
        for report in getattr(workflow_state, 'all_verification_reports', []):
            if report and hasattr(report, 'passed') and not report.passed:
                issues_found.append({
                    "type": getattr(report, 'verification_method', 'unknown'),
                    "details": getattr(report, 'errors', []),
                })
        
        content["issues_found"] = issues_found
        
        artifact = KnowledgeArtifact(
            artifact_id=f"healing_{uuid.uuid4().hex[:16]}",
            artifact_type="self_healing_pattern",
            source_workflow_id=getattr(workflow_state, 'workflow_id', 'unknown'),
            source_stage=5,
            timestamp=datetime.now(),
            confidence=0.75,
            title=f"Self-Healing Pattern: {refinement_count} refinements",
            description=f"Self-healing pattern with {refinement_count} refinement loops",
            content=content,
            tags=["self_healing", f"refinements_{refinement_count}"],
        )
        artifacts.append(artifact)
        
        # Use LLM for enhanced extraction if available
        if self.ace_client:
            try:
                prompt = self.extraction_prompts["self_healing"].format(
                    problem_statement=getattr(workflow_state, 'problem_statement', ''),
                    refinement_count=refinement_count,
                    issues_found=len(issues_found),
                    resolution_strategy="auto" if refinement_count > 0 else "none",
                    success=getattr(workflow_state, 'status', '') == 'completed'
                )
                response = self._call_llm(prompt)
                if response:
                    extracted_data = json.loads(response)
                    content.update({
                        "issue_patterns": extracted_data.get("issue_patterns", []),
                        "resolution_strategies": extracted_data.get("resolution_strategies", []),
                        "refinement_effectiveness": extracted_data.get("refinement_effectiveness", 0.5),
                        "healing_triggers": extracted_data.get("healing_triggers", []),
                        "prevention_recommendations": extracted_data.get("prevention_recommendations", []),
                    })
                    artifact.content = content
            except Exception as e:
                logger.warning(f"LLM self-healing extraction failed: {e}")
        
        return artifacts
    
    # ========== Stage 6: Execution Results & Learning ==========
    
    def _extract_from_stage_6(self, workflow_state: WorkflowState) -> List[KnowledgeArtifact]:
        """
        Extract learning patterns from Stage 6 (Execution Results).
        
        Args:
            workflow_state: The workflow state
            
        Returns:
            List of KnowledgeArtifacts with learning patterns
        """
        artifacts = []
        
        # Only extract from completed workflows
        if getattr(workflow_state, 'status', '') != 'completed':
            return artifacts
        
        # Calculate execution metrics
        start_time = getattr(workflow_state, 'start_time', time.time())
        end_time = getattr(workflow_state, 'end_time', time.time())
        execution_time = end_time - start_time if end_time else 0
        
        # Calculate quality score from final solution
        final_solution = getattr(workflow_state, 'final_solution', None)
        quality_score = 0.0
        if final_solution and hasattr(final_solution, 'quality_score'):
            quality_score = final_solution.quality_score
        
        # Create learning pattern artifact
        content = {
            "execution_time": execution_time,
            "quality_score": quality_score,
            "success": True,
            "sub_problems_solved": len(getattr(workflow_state, 'solved_sub_problem_ids', set())),
            "total_sub_problems": len(getattr(workflow_state, 'sub_problem_solutions', {})),
        }
        
        artifact = KnowledgeArtifact(
            artifact_id=f"learning_{uuid.uuid4().hex[:16]}",
            artifact_type="learning_pattern",
            source_workflow_id=getattr(workflow_state, 'workflow_id', 'unknown'),
            source_stage=6,
            timestamp=datetime.now(),
            confidence=quality_score if quality_score > 0 else 0.8,
            title="Learning Pattern: Successful Execution",
            description="Learning patterns from successful workflow execution",
            content=content,
            tags=["learning", "execution", "success"],
        )
        artifacts.append(artifact)
        
        # Use LLM for enhanced extraction if available
        if self.ace_client:
            try:
                prompt = self.extraction_prompts["learning_patterns"].format(
                    workflow_id=getattr(workflow_state, 'workflow_id', 'unknown'),
                    problem_statement=getattr(workflow_state, 'problem_statement', ''),
                    success=True,
                    execution_time=execution_time,
                    quality_score=quality_score
                )
                response = self._call_llm(prompt)
                if response:
                    extracted_data = json.loads(response)
                    content.update({
                        "success_factors": extracted_data.get("success_factors", []),
                        "improvement_areas": extracted_data.get("improvement_areas", []),
                        "key_learnings": extracted_data.get("key_learnings", []),
                        "reusable_patterns": extracted_data.get("reusable_patterns", []),
                        "adaptation_recommendations": extracted_data.get("adaptation_recommendations", []),
                    })
                    artifact.content = content
            except Exception as e:
                logger.warning(f"LLM learning extraction failed: {e}")
        
        return artifacts
    
    # ========== Helper Methods ==========
    
    def _classify_domain(self, problem_statement: str) -> str:
        """Classify the domain of a problem."""
        problem_lower = problem_statement.lower()
        
        domain_keywords = {
            "algorithms": ["algorithm", "sorting", "searching", "optimization", "graph", "tree"],
            "data_structures": ["array", "list", "tree", "graph", "hash", "queue", "stack"],
            "machine_learning": ["train", "model", "predict", "classify", "regression", "neural"],
            "web_development": ["api", "server", "client", "http", "database", "rest", "endpoint"],
            "system_design": ["scale", "distributed", "architecture", "design", "microservice"],
            "mathematics": ["equation", "theorem", "proof", "calculus", "algebra", "geometry"],
            "physics": ["quantum", "mechanics", "thermodynamics", "electromagnetism", "particle"],
            "chemistry": ["molecule", "reaction", "chemical", "bond", "catalyst", "synthesis"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in problem_lower for kw in keywords):
                return domain
        
        return "general"
    
    def _estimate_complexity(self, problem_statement: str) -> int:
        """Estimate complexity (1-10) of a problem."""
        word_count = len(problem_statement.split())
        
        if word_count < 20:
            return 3
        elif word_count < 50:
            return 5
        elif word_count < 100:
            return 7
        else:
            return 9
    
    def _extract_constraints(self, problem_statement: str) -> List[str]:
        """Extract constraints from a problem statement."""
        constraints = []
        problem_lower = problem_statement.lower()
        
        if "time limit" in problem_lower or "timeout" in problem_lower:
            constraints.append("time_constraint")
        if "memory" in problem_lower:
            constraints.append("memory_constraint")
        if "without using" in problem_lower or "not allowed" in problem_lower:
            constraints.append("restriction")
        if "must" in problem_lower or "require" in problem_lower:
            constraints.append("requirement")
        
        return constraints
    
    def _extract_domain_keywords(self, problem_statement: str, domain: str) -> List[str]:
        """Extract domain-specific keywords from problem statement."""
        problem_lower = problem_statement.lower()
        words = problem_lower.split()
        
        # Simple keyword extraction - in production, use NLP
        keywords = []
        for word in words:
            if len(word) > 4 and word.isalpha():
                keywords.append(word)
        
        return list(set(keywords))[:10]  # Limit to top 10
    
    def _classify_problem_type(self, problem_statement: str) -> str:
        """Classify the type of problem."""
        problem_lower = problem_statement.lower()
        
        if any(word in problem_lower for word in ["implement", "create", "build", "develop"]):
            return "implementation"
        elif any(word in problem_lower for word in ["fix", "debug", "error", "bug"]):
            return "debugging"
        elif any(word in problem_lower for word in ["optimize", "improve", "performance", "speed"]):
            return "optimization"
        elif any(word in problem_lower for word in ["refactor", "restructure", "redesign"]):
            return "refactoring"
        else:
            return "general"
    
    def _extract_code_patterns(self, code: str) -> List[str]:
        """Extract code patterns from solution code."""
        patterns = []
        
        if "def " in code:
            patterns.append("function_definition")
        if "class " in code:
            patterns.append("class_definition")
        if "import " in code or "from " in code:
            patterns.append("module_import")
        if "try:" in code or "except" in code:
            patterns.append("error_handling")
        if "async " in code or "await " in code:
            patterns.append("async_programming")
        if "@" in code:
            patterns.append("decorators")
        if "list comprehension" in code or "[x for x in" in code:
            patterns.append("list_comprehension")
        if "lambda" in code:
            patterns.append("lambda_functions")
        if "with " in code and ":" in code:
            patterns.append("context_managers")
        
        return patterns
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM through ace_client if available."""
        if not self.ace_client:
            return None
        
        try:
            if hasattr(self.ace_client, 'generate'):
                return self.ace_client.generate(prompt)
            elif hasattr(self.ace_client, 'chat'):
                return self.ace_client.chat(prompt)
            elif hasattr(self.ace_client, 'complete'):
                return self.ace_client.complete(prompt)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")

        return None

    def _call_dspy(self, prompt: str, signature=None):
        """
        Call DSPy for enhanced extraction with programmatic prompting.

        Args:
            prompt: Input prompt for DSPy
            signature: Optional DSPy signature for specific task

        Returns:
            DSPy prediction result or None if DSPy is not available
        """
        if not DSPY_AVAILABLE:
            return None

        try:
            # If no specific signature provided, create a generic one
            if signature is None:
                class GenericSignature(dspy.Signature):
                    """Generic extraction signature for various knowledge extraction tasks."""
                    context = dspy.InputField(desc="Context for extraction")
                    question = dspy.InputField(desc="Specific extraction question")
                    answer = dspy.OutputField(desc="Extraction result in structured format")

                # Create a predictor with the signature
                predictor = dspy.Predict(GenericSignature)

                # Run the prediction
                result = predictor(context=prompt, question="Extract relevant information")
                return result.answer
            else:
                # Use provided signature
                predictor = dspy.Predict(signature)
                result = predictor(**signature)
                return result

        except Exception as e:
            logger.warning(f"DSPy call failed: {e}")
            return None

    def _create_dspy_solution_pattern_signature(self):
        """
        Create a DSPy signature for extracting solution patterns.

        Returns:
            DSPy signature for solution pattern extraction
        """
        if not DSPY_AVAILABLE:
            return None

        class SolutionPatternSignature(dspy.Signature):
            """Extract solution patterns from workflow execution data."""
            problem_statement = dspy.InputField(desc="Original problem statement")
            decomposition_strategy = dspy.InputField(desc="Decomposition strategy used (ROMA/MAKER/MDAP)")
            final_solution = dspy.InputField(desc="Final solution code/text")
            success = dspy.InputField(desc="Whether the solution was successful (true/false)")

            problem_characteristics = dspy.OutputField(desc="List of problem characteristics (domain, complexity, constraints)")
            solution_approach = dspy.OutputField(desc="High-level solution approach")
            code_patterns = dspy.OutputField(desc="Key code patterns used")
            optimization_techniques = dspy.OutputField(desc="Optimization techniques applied")
            typical_refinements = dspy.OutputField(desc="Typical refinements needed")
            domain = dspy.OutputField(desc="Problem domain (algorithms, data_structures, etc.)")
            complexity = dspy.OutputField(desc="Complexity rating (1-10)")

        return SolutionPatternSignature

    def _create_dspy_decomposition_signature(self):
        """
        Create a DSPy signature for extracting decomposition strategies.

        Returns:
            DSPy signature for decomposition strategy extraction
        """
        if not DSPY_AVAILABLE:
            return None

        class DecompositionSignature(dspy.Signature):
            """Extract decomposition strategies from workflow execution data."""
            problem_statement = dspy.InputField(desc="Original problem statement")
            strategy = dspy.InputField(desc="Decomposition strategy used")
            framework = dspy.InputField(desc="Framework used (ROMA/MAKER/MDAP)")
            num_sub_problems = dspy.InputField(desc="Number of sub-problems created")
            success = dspy.InputField(desc="Whether the decomposition was successful (true/false)")

            strategy_insights = dspy.OutputField(desc="Key insights about the decomposition strategy")
            effectiveness_score = dspy.OutputField(desc="Effectiveness score (0.0-1.0)")
            improvement_suggestions = dspy.OutputField(desc="Suggestions for improving the strategy")

        return DecompositionSignature


class DSPySolutionPatternExtractor:
    """
    DSPy-based extractor for solution patterns from successful solutions.

    This class leverages DSPy's programmatic prompting capabilities to extract
    solution patterns from successful solution attempts with improved consistency
    and performance compared to traditional prompting approaches.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", ace_client: Optional[Any] = None):
        """
        Initialize the DSPy-based solution pattern extractor.

        Args:
            model_name: Name of the LLM to use with DSPy
            ace_client: Optional ACE client for fallback
        """
        self.model_name = model_name
        self.ace_client = ace_client

        if DSPY_AVAILABLE:
            # Set up DSPy with the specified model
            try:
                # Use LiteLLM for flexibility with different models
                dspy.settings.configure(lm=dspy.LM(model=model_name))
            except Exception as e:
                logger.warning(f"Failed to configure DSPy with model {model_name}: {e}")
                # Try to configure with a default model
                try:
                    dspy.settings.configure(lm=dspy.LM(model="gpt-4o-mini"))
                except:
                    logger.warning("Could not configure DSPy LLM, will use fallback methods")
        else:
            logger.warning("DSPy not available, using fallback extraction methods")

    def extract_solution_patterns(self, solutions: List[SolutionAttempt]) -> List[SolutionPatternArtifact]:
        """
        Extract solution patterns from a list of solutions using DSPy.

        Args:
            solutions: List of solution attempts to analyze

        Returns:
            List of SolutionPatternArtifacts
        """
        if not DSPY_AVAILABLE:
            # Fallback to basic extraction
            return self._extract_with_fallback(solutions)

        patterns = []
        for solution in solutions:
            if not solution:
                continue

            # Filter for high-quality solutions (>0.8 score)
            quality_score = getattr(solution, 'quality_score', 0.0)
            if quality_score < 0.8:
                continue

            pattern = self._dspy_analyze_solution_pattern(solution)
            if pattern:
                patterns.append(pattern)

        return patterns

    def _dspy_analyze_solution_pattern(self, solution: SolutionAttempt) -> Optional[SolutionPatternArtifact]:
        """
        Analyze a single solution to extract its pattern using DSPy.

        Args:
            solution: The solution attempt to analyze

        Returns:
            SolutionPatternArtifact if analysis successful
        """
        if not DSPY_AVAILABLE:
            return None

        try:
            # Define the DSPy signature for solution pattern extraction
            class SolutionPatternSignature(dspy.Signature):
                """Extract solution patterns from workflow execution data."""
                problem_statement = dspy.InputField(desc="Original problem statement")
                decomposition_strategy = dspy.InputField(desc="Decomposition strategy used (ROMA/MAKER/MDAP)")
                final_solution = dspy.InputField(desc="Final solution code/text")
                success = dspy.InputField(desc="Whether the solution was successful (true/false)")

                problem_characteristics = dspy.OutputField(desc="List of problem characteristics (domain, complexity, constraints)")
                solution_approach = dspy.OutputField(desc="High-level solution approach")
                code_patterns = dspy.OutputField(desc="Key code patterns used")
                optimization_techniques = dspy.OutputField(desc="Optimization techniques applied")
                typical_refinements = dspy.OutputField(desc="Typical refinements needed")
                domain = dspy.OutputField(desc="Problem domain (algorithms, data_structures, etc.)")
                complexity = dspy.OutputField(desc="Complexity rating (1-10)")

            # Create predictor
            predictor = dspy.Predict(SolutionPatternSignature)

            # Prepare inputs
            problem_statement = getattr(solution, 'problem_statement', 'Unknown problem')
            decomposition_strategy = getattr(solution, 'decomposition_strategy', 'Unknown')
            final_code = getattr(solution, 'final_code', '')
            success = str(getattr(solution, 'is_successful', False))

            # Run prediction
            result = predictor(
                problem_statement=problem_statement,
                decomposition_strategy=decomposition_strategy,
                final_solution=final_code,
                success=success
            )

            # Create the artifact from DSPy results
            content = {
                "code_language": getattr(solution, 'code_language', 'python'),
                "solution_quality": getattr(solution, 'quality_score', 0.0),
                "approach": getattr(solution, 'approach_description', ''),
                "complexity": getattr(solution, 'complexity_score', 5),
                "dspy_analysis": {
                    "problem_characteristics": result.problem_characteristics,
                    "solution_approach": result.solution_approach,
                    "code_patterns": result.code_patterns,
                    "optimization_techniques": result.optimization_techniques,
                    "typical_refinements": result.typical_refinements,
                    "domain": result.domain,
                    "complexity": result.complexity
                }
            }

            artifact = SolutionPatternArtifact(
                artifact_id=f"dspy_solution_pattern_{uuid.uuid4().hex[:16]}",
                artifact_type="solution_pattern",
                source_workflow_id=getattr(solution, 'workflow_id', 'unknown'),
                source_stage=3,
                timestamp=datetime.now(),
                confidence=getattr(solution, 'quality_score', 0.8),
                title=f"DSPy Solution Pattern: {getattr(solution, 'approach_type', 'General')}",
                description=getattr(solution, 'approach_description', 'Solution pattern extracted using DSPy'),
                content=content,
                pattern_category=getattr(solution, 'approach_type', 'general'),
                problem_domains=[result.domain if hasattr(result, 'domain') and result.domain else getattr(solution, 'domain', 'general')],
                approach_signature={
                    "quality_score": getattr(solution, 'quality_score', 0.0),
                    "complexity": int(result.complexity) if hasattr(result, 'complexity') and result.complexity.isdigit() else getattr(solution, 'complexity_score', 5),
                    "language": getattr(solution, 'code_language', 'python'),
                },
                success_rate=1.0 if getattr(solution, 'is_successful', False) else 0.5,
                avg_execution_time=getattr(solution, 'execution_time', 0.0),
                tags=["dspy_solution_pattern", getattr(solution, 'code_language', 'python'),
                      getattr(solution, 'approach_type', 'general'), "dspy_enhanced"],
            )

            return artifact

        except Exception as e:
            logger.warning(f"DSPy solution pattern analysis failed: {e}")
            # Fallback to basic extraction
            if self.ace_client:
                return self._extract_with_fallback([solution])[0] if self._extract_with_fallback([solution]) else None
            return None

    def _extract_with_fallback(self, solutions: List[SolutionAttempt]) -> List[SolutionPatternArtifact]:
        """
        Fallback extraction method when DSPy is not available.

        Args:
            solutions: List of solution attempts to analyze

        Returns:
            List of SolutionPatternArtifacts
        """
        # Simple fallback extraction
        patterns = []
        for solution in solutions:
            if not solution:
                continue

            quality_score = getattr(solution, 'quality_score', 0.0)
            if quality_score < 0.8:
                continue

            content = {
                "code_language": getattr(solution, 'code_language', 'python'),
                "solution_quality": quality_score,
                "approach": getattr(solution, 'approach_description', ''),
                "complexity": getattr(solution, 'complexity_score', 5),
                "fallback_extraction": True
            }

            artifact = SolutionPatternArtifact(
                artifact_id=f"fallback_solution_pattern_{uuid.uuid4().hex[:16]}",
                artifact_type="solution_pattern",
                source_workflow_id=getattr(solution, 'workflow_id', 'unknown'),
                source_stage=3,
                timestamp=datetime.now(),
                confidence=quality_score,
                title=f"Fallback Solution Pattern: {getattr(solution, 'approach_type', 'General')}",
                description=getattr(solution, 'approach_description', 'Solution pattern extracted using fallback method'),
                content=content,
                pattern_category=getattr(solution, 'approach_type', 'general'),
                problem_domains=[getattr(solution, 'domain', 'general')],
                approach_signature={
                    "quality_score": quality_score,
                    "complexity": getattr(solution, 'complexity_score', 5),
                    "language": getattr(solution, 'code_language', 'python'),
                },
                success_rate=1.0 if getattr(solution, 'is_successful', False) else 0.5,
                avg_execution_time=getattr(solution, 'execution_time', 0.0),
                tags=["fallback_solution_pattern", getattr(solution, 'code_language', 'python'),
                      getattr(solution, 'approach_type', 'general')],
            )

            patterns.append(artifact)

        return patterns


class SolutionPatternExtractor:
    """
    Specialized extractor for solution patterns from successful solutions.
    
    This class focuses on extracting reusable solution patterns that can be
    applied to similar problems in the future.
    
    Attributes:
        ace_client: Optional ACE client for LLM-based extraction
    """
    
    def __init__(self, ace_client: Optional[Any] = None):
        """
        Initialize the solution pattern extractor.
        
        Args:
            ace_client: Optional ACE client for LLM-based extraction
        """
        self.ace_client = ace_client
    
    def extract_patterns(self, solutions: List[SolutionAttempt]) -> List[SolutionPatternArtifact]:
        """
        Extract solution patterns from a list of solutions.
        
        Args:
            solutions: List of solution attempts to analyze
            
        Returns:
            List of SolutionPatternArtifacts
        """
        patterns = []
        
        for solution in solutions:
            if not solution:
                continue
            
            # Filter for high-quality solutions (>0.8 score)
            quality_score = getattr(solution, 'quality_score', 0.0)
            if quality_score < 0.8:
                continue
            
            pattern = self._analyze_solution_pattern(solution)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_solution_pattern(self, solution: SolutionAttempt) -> Optional[SolutionPatternArtifact]:
        """
        Analyze a single solution to extract its pattern.

        Args:
            solution: The solution attempt to analyze

        Returns:
            SolutionPatternArtifact if analysis successful
        """
        try:
            # Extract pattern characteristics
            content = {
                "code_language": getattr(solution, 'code_language', 'python'),
                "solution_quality": getattr(solution, 'quality_score', 0.0),
                "approach": getattr(solution, 'approach_description', ''),
                "complexity": getattr(solution, 'complexity_score', 5),
            }

            # Extract code patterns from final code
            final_code = getattr(solution, 'final_code', '')
            if final_code:
                content["code_patterns"] = self._extract_code_patterns_from_code(final_code)
                content["code_structure"] = self._analyze_code_structure(final_code)

            # Use DSPy for enhanced extraction if available
            if DSPY_AVAILABLE:
                dspy_signature = self._create_dspy_solution_pattern_signature()
                if dspy_signature:
                    try:
                        # Prepare input for DSPy
                        problem_statement = getattr(solution, 'problem_statement', 'Unknown problem')
                        decomposition_strategy = getattr(solution, 'decomposition_strategy', 'Unknown')
                        success = str(getattr(solution, 'is_successful', False))

                        # Call DSPy for enhanced analysis
                        dspy_result = self._call_dspy(
                            prompt=f"Problem: {problem_statement}\nDecomposition: {decomposition_strategy}\nCode: {final_code}\nSuccess: {success}",
                            signature=dspy_signature()
                        )

                        # Integrate DSPy results into content if available
                        if dspy_result:
                            content["dspy_analysis"] = dspy_result

                    except Exception as e:
                        logger.warning(f"DSPy solution pattern analysis failed: {e}")

            # Create the artifact
            artifact = SolutionPatternArtifact(
                artifact_id=f"solution_pattern_{uuid.uuid4().hex[:16]}",
                artifact_type="solution_pattern",
                source_workflow_id=getattr(solution, 'workflow_id', 'unknown'),
                source_stage=3,
                timestamp=datetime.now(),
                confidence=getattr(solution, 'quality_score', 0.8),
                title=f"Solution Pattern: {getattr(solution, 'approach_type', 'General')}",
                description=getattr(solution, 'approach_description', 'Solution pattern extracted from successful execution'),
                content=content,
                pattern_category=getattr(solution, 'approach_type', 'general'),
                problem_domains=[getattr(solution, 'domain', 'general')],
                approach_signature={
                    "quality_score": getattr(solution, 'quality_score', 0.0),
                    "complexity": getattr(solution, 'complexity_score', 5),
                    "language": getattr(solution, 'code_language', 'python'),
                },
                success_rate=1.0 if getattr(solution, 'is_successful', False) else 0.5,
                avg_execution_time=getattr(solution, 'execution_time', 0.0),
                tags=["solution_pattern", getattr(solution, 'code_language', 'python'),
                      getattr(solution, 'approach_type', 'general')],
            )

            return artifact

        except Exception as e:
            logger.warning(f"Failed to analyze solution pattern: {e}")
            return None
    
    def _extract_code_patterns_from_code(self, code: str) -> List[str]:
        """Extract code patterns from solution code."""
        patterns = []
        
        if "def " in code:
            patterns.append("function_definition")
        if "class " in code:
            patterns.append("class_definition")
        if "import " in code or "from " in code:
            patterns.append("module_import")
        if "try:" in code or "except" in code:
            patterns.append("error_handling")
        if "async " in code or "await " in code:
            patterns.append("async_programming")
        if "@" in code:
            patterns.append("decorators")
        if "[x for x in" in code:
            patterns.append("list_comprehension")
        if "lambda" in code:
            patterns.append("lambda_functions")
        if "with " in code and ":" in code:
            patterns.append("context_managers")
        if "if __name__ ==" in code:
            patterns.append("main_guard")
        
        return patterns
    
    def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze the structure of code."""
        lines = code.split('\n')
        return {
            "total_lines": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "function_count": code.count("def "),
            "class_count": code.count("class "),
            "import_count": code.count("import ") + code.count("from "),
            "has_docstrings": '"""' in code or "'''" in code,
            "has_type_hints": ": " in code and "->" in code,
        }


class DSPyDecompositionStrategyExtractor:
    """
    DSPy-based extractor for decomposition strategies from workflow execution results.

    This class leverages DSPy's programmatic prompting capabilities to extract
    decomposition strategies from workflow execution results with improved consistency
    and performance compared to traditional prompting approaches.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", ace_client: Optional[Any] = None):
        """
        Initialize the DSPy-based decomposition strategy extractor.

        Args:
            model_name: Name of the LLM to use with DSPy
            ace_client: Optional ACE client for fallback
        """
        self.model_name = model_name
        self.ace_client = ace_client

        if DSPY_AVAILABLE:
            # Set up DSPy with the specified model
            try:
                dspy.settings.configure(lm=dspy.LM(model=model_name))
            except Exception as e:
                logger.warning(f"Failed to configure DSPy with model {model_name}: {e}")
                # Try to configure with a default model
                try:
                    dspy.settings.configure(lm=dspy.LM(model="gpt-4o-mini"))
                except:
                    logger.warning("Could not configure DSPy LLM, will use fallback methods")
        else:
            logger.warning("DSPy not available, using fallback extraction methods")

    def extract_strategies(self, execution_results: List[Dict[str, Any]]) -> List[KnowledgeArtifact]:
        """
        Extract decomposition strategies from execution results using DSPy.

        Args:
            execution_results: List of execution results to analyze

        Returns:
            List of KnowledgeArtifacts representing decomposition strategies
        """
        if not DSPY_AVAILABLE:
            # Fallback to basic extraction
            return self._extract_with_fallback(execution_results)

        strategies = []
        for result in execution_results:
            if not result:
                continue

            strategy = self._dspy_analyze_decomposition_strategy(result)
            if strategy:
                strategies.append(strategy)

        return strategies

    def _dspy_analyze_decomposition_strategy(self, result: Dict[str, Any]) -> Optional[KnowledgeArtifact]:
        """
        Analyze a single execution result to extract its decomposition strategy using DSPy.

        Args:
            result: The execution result to analyze

        Returns:
            KnowledgeArtifact if analysis successful
        """
        if not DSPY_AVAILABLE:
            return None

        try:
            # Define the DSPy signature for decomposition strategy extraction
            class DecompositionSignature(dspy.Signature):
                """Extract decomposition strategies from workflow execution data."""
                problem_statement = dspy.InputField(desc="Original problem statement")
                strategy = dspy.InputField(desc="Decomposition strategy used")
                framework = dspy.InputField(desc="Framework used (ROMA/MAKER/MDAP)")
                num_sub_problems = dspy.InputField(desc="Number of sub-problems created")
                success = dspy.InputField(desc="Whether the decomposition was successful (true/false)")

                strategy_insights = dspy.OutputField(desc="Key insights about the decomposition strategy")
                effectiveness_score = dspy.OutputField(desc="Effectiveness score (0.0-1.0)")
                improvement_suggestions = dspy.OutputField(desc="Suggestions for improving the strategy")

            # Create predictor
            predictor = dspy.Predict(DecompositionSignature)

            # Prepare inputs
            problem_statement = result.get('problem_statement', 'Unknown problem')
            strategy = result.get('strategy', 'Unknown')
            framework = result.get('framework', 'Unknown')
            num_sub_problems = str(result.get('num_sub_problems', 0))
            success = str(result.get('success', False))

            # Run prediction
            dspy_result = predictor(
                problem_statement=problem_statement,
                strategy=strategy,
                framework=framework,
                num_sub_problems=num_sub_problems,
                success=success
            )

            # Create the artifact from DSPy results
            content = {
                "dspy_analysis": {
                    "strategy_insights": dspy_result.strategy_insights,
                    "effectiveness_score": dspy_result.effectiveness_score,
                    "improvement_suggestions": dspy_result.improvement_suggestions
                },
                "raw_result": result
            }

            artifact = KnowledgeArtifact(
                artifact_id=f"dspy_decomposition_strategy_{uuid.uuid4().hex[:16]}",
                artifact_type="decomposition_strategy",
                source_workflow_id=result.get('workflow_id', 'unknown'),
                source_stage=3,
                timestamp=datetime.now(),
                confidence=float(dspy_result.effectiveness_score) if dspy_result.effectiveness_score.replace('.', '').isdigit() else 0.8,
                title=f"DSPy Decomposition Strategy: {strategy}",
                description=f"Decomposition strategy extracted using DSPy: {dspy_result.strategy_insights}",
                content=content,
                tags=["dspy_decomposition", framework, "dspy_enhanced"]
            )

            return artifact

        except Exception as e:
            logger.warning(f"DSPy decomposition strategy analysis failed: {e}")
            # Fallback to basic extraction
            if self.ace_client:
                return self._extract_with_fallback([result])[0] if self._extract_with_fallback([result]) else None
            return None

    def _extract_with_fallback(self, execution_results: List[Dict[str, Any]]) -> List[KnowledgeArtifact]:
        """
        Fallback extraction method when DSPy is not available.

        Args:
            execution_results: List of execution results to analyze

        Returns:
            List of KnowledgeArtifacts
        """
        # Simple fallback extraction
        strategies = []
        for result in execution_results:
            if not result:
                continue

            content = {
                "raw_result": result,
                "fallback_extraction": True
            }

            artifact = KnowledgeArtifact(
                artifact_id=f"fallback_decomposition_strategy_{uuid.uuid4().hex[:16]}",
                artifact_type="decomposition_strategy",
                source_workflow_id=result.get('workflow_id', 'unknown'),
                source_stage=3,
                timestamp=datetime.now(),
                confidence=0.8,
                title=f"Fallback Decomposition Strategy: {result.get('strategy', 'Unknown')}",
                description="Decomposition strategy extracted using fallback method",
                content=content,
                tags=["fallback_decomposition", "fallback_extraction"],
            )

            strategies.append(artifact)

        return strategies


class DecompositionStrategyExtractor:
    """
    Specialized extractor for decomposition strategies from execution results.
    
    This class focuses on extracting reusable decomposition strategies that can be
    applied to similar problems in the future.
    
    Attributes:
        ace_client: Optional ACE client for LLM-based extraction
    """
    
    def __init__(self, ace_client: Optional[Any] = None):
        """
        Initialize the decomposition strategy extractor.
        
        Args:
            ace_client: Optional ACE client for LLM-based extraction
        """
        self.ace_client = ace_client
    
    def extract_strategies(self, execution_results: List[Dict[str, Any]]) -> List[KnowledgeArtifact]:
        """
        Extract decomposition strategies from execution results.
        
        Args:
            execution_results: List of execution results to analyze
            
        Returns:
            List of KnowledgeArtifacts containing decomposition strategies
        """
        artifacts = []
        
        for result in execution_results:
            if not result:
                continue
            
            artifact = self._analyze_decomposition_strategy(result)
            if artifact:
                artifacts.append(artifact)
        
        return artifacts
    
    def _analyze_decomposition_strategy(self, result: Dict[str, Any]) -> Optional[KnowledgeArtifact]:
        """
        Analyze a single execution result to extract decomposition strategy.
        
        Args:
            result: The execution result to analyze
            
        Returns:
            KnowledgeArtifact if analysis successful
        """
        try:
            decomposition_plan = result.get('decomposition_plan')
            if not decomposition_plan:
                return None
            
            # Extract strategy characteristics
            framework = getattr(decomposition_plan, 'framework', 'unknown')
            strategy = getattr(decomposition_plan, 'decomposition_method', 'unknown')
            
            content = {
                "framework": framework,
                "strategy": strategy,
                "num_sub_problems": len(getattr(decomposition_plan, 'sub_problems', [])),
                "success": result.get('success', False),
                "execution_time": result.get('execution_time', 0),
            }
            
            # Extract sub-problem characteristics
            sub_problems = getattr(decomposition_plan, 'sub_problems', [])
            content["sub_problem_types"] = [
                getattr(sp, 'type', 'unknown') for sp in sub_problems
            ]
            
            # Extract dependencies if available
            dependencies = getattr(decomposition_plan, 'dependencies', [])
            content["dependency_pattern"] = self._analyze_dependency_pattern(dependencies)
            
            # Create the artifact
            artifact = KnowledgeArtifact(
                artifact_id=f"decomp_strategy_{uuid.uuid4().hex[:16]}",
                artifact_type="decomposition_strategy",
                source_workflow_id=result.get('workflow_id', 'unknown'),
                source_stage=1,
                timestamp=datetime.now(),
                confidence=0.85 if result.get('success') else 0.6,
                title=f"Decomposition Strategy: {framework}",
                description=f"Decomposition strategy using {framework} with {strategy} method",
                content=content,
                tags=["decomposition", framework, strategy],
            )
            
            return artifact
            
        except Exception as e:
            logger.warning(f"Failed to analyze decomposition strategy: {e}")
            return None
    
    def _analyze_dependency_pattern(self, dependencies: List[Any]) -> Dict[str, Any]:
        """Analyze the dependency pattern of sub-problems."""
        if not dependencies:
            return {"type": "independent", "count": 0}
        
        # Simple analysis - can be enhanced with graph analysis
        return {
            "type": "sequential" if len(dependencies) > 0 else "independent",
            "count": len(dependencies),
            "chains": len(set(str(d) for d in dependencies)),
        }


# ========== Convenience Functions ==========

def extract_knowledge_from_workflow(workflow_state: WorkflowState, 
                                    knowledge_engine: Optional[Any] = None,
                                    ace_client: Optional[Any] = None) -> List[KnowledgeArtifact]:
    """
    Convenience function to extract all knowledge from a workflow.
    
    Args:
        workflow_state: The workflow state to extract from
        knowledge_engine: Optional knowledge engine for enhanced extraction
        ace_client: Optional ACE client for LLM-based extraction
        
    Returns:
        List of extracted KnowledgeArtifacts
    """
    extractor = WorkflowKnowledgeExtractor(knowledge_engine, ace_client)
    return extractor.extract_from_workflow(workflow_state)


def extract_solution_patterns(solutions: List[SolutionAttempt],
                              ace_client: Optional[Any] = None) -> List[SolutionPatternArtifact]:
    """
    Convenience function to extract solution patterns from solutions.
    
    Args:
        solutions: List of solution attempts
        ace_client: Optional ACE client for LLM-based extraction
        
    Returns:
        List of SolutionPatternArtifacts
    """
    extractor = SolutionPatternExtractor(ace_client)
    return extractor.extract_patterns(solutions)


def extract_decomposition_strategies(execution_results: List[Dict[str, Any]],
                                     ace_client: Optional[Any] = None) -> List[KnowledgeArtifact]:
    """
    Convenience function to extract decomposition strategies.
    
    Args:
        execution_results: List of execution results
        ace_client: Optional ACE client for LLM-based extraction
        
    Returns:
        List of KnowledgeArtifacts containing decomposition strategies
    """
    extractor = DecompositionStrategyExtractor(ace_client)
    return extractor.extract_strategies(execution_results)


def extract_solution_patterns_with_dspy(solutions: List[SolutionAttempt],
                                        model_name: str = "gpt-4o-mini",
                                        ace_client: Optional[Any] = None) -> List[SolutionPatternArtifact]:
    """
    Convenience function to extract solution patterns from solutions using DSPy.

    Args:
        solutions: List of solution attempts
        model_name: Name of the LLM to use with DSPy
        ace_client: Optional ACE client for fallback

    Returns:
        List of SolutionPatternArtifacts
    """
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available, falling back to regular extraction")
        return extract_solution_patterns(solutions, ace_client)

    extractor = DSPySolutionPatternExtractor(model_name=model_name, ace_client=ace_client)
    return extractor.extract_solution_patterns(solutions)


def extract_decomposition_strategies_with_dspy(execution_results: List[Dict[str, Any]],
                                               model_name: str = "gpt-4o-mini",
                                               ace_client: Optional[Any] = None) -> List[KnowledgeArtifact]:
    """
    Convenience function to extract decomposition strategies using DSPy.

    Args:
        execution_results: List of execution results
        model_name: Name of the LLM to use with DSPy
        ace_client: Optional ACE client for fallback

    Returns:
        List of KnowledgeArtifacts containing decomposition strategies
    """
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available, falling back to regular extraction")
        return extract_decomposition_strategies(execution_results, ace_client)

    extractor = DSPyDecompositionStrategyExtractor(model_name=model_name, ace_client=ace_client)
    return extractor.extract_strategies(execution_results)

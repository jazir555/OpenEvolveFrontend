"""
Workflow Knowledge Extractor - Stage 6 Knowledge Extraction

This module extracts knowledge artifacts from workflow executions across all stages.
It uses LLM-based semantic extraction to identify patterns, team performance insights,
and gauntlet effectiveness metrics.

Enhanced with OneKE integration for schema-guided domain knowledge extraction.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
import json
import asyncio

from workflow_structures import (
    WorkflowState,
    SolutionPatternArtifact,
    TeamPerformanceArtifact,
    GauntletEffectivenessArtifact,
    KnowledgeArtifactManager,
    Team,
    GauntletDefinition,
    SolutionAttempt,
    CritiqueReport,
    VerificationReport,
    DecompositionPlan,
)


class WorkflowKnowledgeExtractor:
    """
    Extracts knowledge artifacts from workflow executions.

    This class analyzes workflow data from all stages to extract:
    - Solution patterns (successful problem-solving approaches)
    - Team performance insights (which teams work best for which problems)
    - Gauntlet effectiveness (which quality checks work best)

    Attributes:
        artifact_manager: Manager for storing/retrieving artifacts
        llm_client: Optional LLM client for semantic extraction
        extraction_prompts: Dictionary of prompts for each extraction type
    """

    def __init__(self, db_path: str = "./knowledge_artifacts.db", llm_client: Optional[Any] = None,
                 use_oneke: bool = False):
        """
        Initialize the knowledge extractor.

        Args:
            db_path: Path to artifact database
            llm_client: Optional LLM client for advanced extraction
            use_oneke: Whether to use OneKE for schema-guided extraction (default: False)
        """
        self.artifact_manager = KnowledgeArtifactManager(db_path)
        self.llm_client = llm_client
        self.use_oneke = use_oneke
        self.oneke_bridge = None
        self.extraction_prompts = self._init_extraction_prompts()

        # Initialize OneKE bridge if enabled
        if use_oneke:
            try:
                # Import here to avoid hard dependency
                from integrations.oneke import OneKEBridge
                self.oneke_bridge = OneKEBridge()
                # Don't auto-initialize - require explicit call
                self._oneke_initialized = False
                logger.info("OneKE bridge created. Call ensure_oneke_initialized() before use.")
            except ImportError:
                print("OneKE not available. Install with: pip install integrations/oneke")
                self.use_oneke = False
                self.oneke_bridge = None
                self._oneke_initialized = False

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
        }

    # ========== Stage 0: Problem Definition ==========

    def extract_from_problem_definition(self, workflow: WorkflowState) -> List[str]:
        """
        Extract insights from Stage 0 (Problem Definition).

        Args:
            workflow: The workflow state

        Returns:
            List of problem characteristics
        """
        characteristics = []

        # Extract domain
        domain = self._classify_domain(workflow.problem_statement)
        if domain:
            characteristics.append(f"domain: {domain}")

        # Extract complexity
        complexity = self._estimate_complexity(workflow.problem_statement)
        characteristics.append(f"complexity: {complexity}")

        # Extract constraints
        constraints = self._extract_constraints(workflow.problem_statement)
        characteristics.extend([f"constraint: {c}" for c in constraints])

        return characteristics

    # ========== Stage 1: Decomposition ==========

    def extract_from_decomposition(self, workflow: WorkflowState) -> Optional[SolutionPatternArtifact]:
        """
        Extract solution patterns from Stage 1 (Decomposition).

        Args:
            workflow: The workflow state

        Returns:
            SolutionPatternArtifact if extraction successful
        """
        if not workflow.decomposition_plan:
            return None

        # Extract decomposition strategy
        strategy = workflow.decomposition_plan.decomposition_method if hasattr(workflow.decomposition_plan, 'decomposition_method') else "unknown"

        # Use LLM for detailed extraction
        if self.llm_client:
            prompt = self.extraction_prompts["solution_pattern"].format(
                problem_statement=workflow.problem_statement,
                decomposition_strategy=strategy,
                final_solution="",
                success=False
            )
            try:
                response = self.llm_client.generate(prompt)
                extracted_data = json.loads(response)

                artifact = SolutionPatternArtifact(
                    artifact_id=f"pattern_{uuid.uuid4().hex[:16]}",
                    source_workflow_id=workflow.workflow_id,
                    problem_characteristics=extracted_data.get("problem_characteristics", []),
                    solution_approach=extracted_data.get("solution_approach", ""),
                    decomposition_strategy=strategy,
                    code_patterns=extracted_data.get("code_patterns", []),
                    optimization_techniques=extracted_data.get("optimization_techniques", []),
                    typical_refinements=extracted_data.get("typical_refinements", []),
                    domain=extracted_data.get("domain", ""),
                    complexity=extracted_data.get("complexity", 5),
                )
                artifact.pattern_signature = artifact.calculate_signature()
                return artifact
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"LLM extraction failed: {e}")

        # Fallback: Create basic artifact
        artifact = SolutionPatternArtifact(
            artifact_id=f"pattern_{uuid.uuid4().hex[:16]}",
            source_workflow_id=workflow.workflow_id,
            decomposition_strategy=strategy,
            problem_characteristics=self.extract_from_problem_definition(workflow),
        )
        artifact.pattern_signature = artifact.calculate_signature()
        return artifact

    def extract_decomposition_strategy(self, workflow: WorkflowState) -> Optional[SolutionPatternArtifact]:
        """
        Extract decomposition strategy (ROMA/MAKER/MDAP) from workflow.

        This is an alias for extract_from_decomposition() to satisfy MASTER_TASKLIST requirements.

        Args:
            workflow: The workflow state

        Returns:
            SolutionPatternArtifact with decomposition strategy information
        """
        return self.extract_from_decomposition(workflow)

    # ========== Stage 3: Code Generation ==========

    def extract_from_code_generation(self, workflow: WorkflowState) -> List[str]:
        """
        Extract code patterns from Stage 3 (Code Generation).

        Args:
            workflow: The workflow state

        Returns:
            List of code patterns
        """
        patterns = []

        for sub_problem_id, solution in (workflow.sub_problem_solutions or {}).items():
            if not solution:
                logger.warning(f"Skipping None solution for {sub_problem_id}")
                continue

            if solution.final_code:
                # Extract language
                patterns.append(f"language: {solution.code_language if hasattr(solution, 'code_language') else 'python'}")

                # Extract common patterns from code
                code = solution.final_code
                if "def " in code:
                    patterns.append("pattern: function_definition")
                if "class " in code:
                    patterns.append("pattern: class_definition")
                if "import " in code:
                    patterns.append("pattern: module_import")
                if "try:" in code or "except" in code:
                    patterns.append("pattern: error_handling")

        return list(set(patterns))  # Deduplicate

    # ========== Stage 5: Quality Assessment ==========

    def extract_from_quality_assessment(self, workflow: WorkflowState) -> Dict[str, Any]:
        """
        Extract quality insights from Stage 5 (Quality Assessment).

        Args:
            workflow: The workflow state

        Returns:
            Dictionary of quality insights
        """
        insights = {
            "total_critiques": len(workflow.all_critique_reports),
            "total_verifications": len(workflow.all_verification_reports),
            "critique_types": [],
            "verification_methods": [],
        }

        for critique in (workflow.all_critique_reports or []):
            if not critique:
                continue
            insights["critique_types"].append(critique.critique_type if hasattr(critique, 'critique_type') else "general")

        for verification in workflow.all_verification_reports:
            if verification:
                insights["verification_methods"].append(str(verification.verification_method) if hasattr(verification, 'verification_method') else "standard")

        return insights

    # ========== Stage 6: Execution Results ==========

    def extract_from_execution_results(self, workflow: WorkflowState) -> List[SolutionPatternArtifact]:
        """
        Extract solution patterns from Stage 6 (Execution Results).

        Args:
            workflow: The workflow state

        Returns:
            List of SolutionPatternArtifacts
        """
        artifacts = []

        # Check if workflow was successful
        success = workflow.final_solution is not None and workflow.status == "completed"

        if success and workflow.final_solution:
            # Create pattern from successful solution
            artifact = SolutionPatternArtifact(
                artifact_id=f"pattern_{uuid.uuid4().hex[:16]}",
                source_workflow_id=workflow.workflow_id,
                success_rate=1.0,
                confidence=0.9,
                solution_approach="Final solution from " + workflow.workflow_id,
                problem_characteristics=self.extract_from_problem_definition(workflow),
                code_patterns=self.extract_from_code_generation(workflow),
            )

            # Extract decomposition strategy
            if workflow.decomposition_plan:
                artifact.decomposition_strategy = workflow.decomposition_plan.decomposition_method if hasattr(workflow.decomposition_plan, 'decomposition_method') else "unknown"

            artifact.pattern_signature = artifact.calculate_signature()
            artifacts.append(artifact)

        return artifacts

    # ========== Solution Pattern Extraction ==========

    def extract_solution_patterns(self, workflow: WorkflowState) -> List[SolutionPatternArtifact]:
        """
        Extract all solution patterns from a workflow.

        Args:
            workflow: The workflow state

        Returns:
            List of SolutionPatternArtifacts
        """
        patterns = []

        # Extract from decomposition
        decomposition_pattern = self.extract_from_decomposition(workflow)
        if decomposition_pattern:
            patterns.append(decomposition_pattern)

        # Extract from execution results
        execution_patterns = self.extract_from_execution_results(workflow)
        patterns.extend(execution_patterns)

        return patterns

    # ========== Team Performance Extraction ==========

    def extract_team_performance(self, workflow: WorkflowState) -> List[TeamPerformanceArtifact]:
        """
        Extract team performance insights from a workflow.

        Args:
            workflow: The workflow state

        Returns:
            List of TeamPerformanceArtifacts
        """
        artifacts = []

        # Extract solver team performance
        if workflow.solver_team:
            artifact = self._analyze_team_performance(workflow, workflow.solver_team, "solver")
            if artifact:
                artifacts.append(artifact)

        # Extract evaluator team performance (Gold Team)
        if workflow.final_gold_gauntlet:
            gold_team = workflow.final_gold_gauntlet.team if hasattr(workflow.final_gold_gauntlet, 'team') else None
            if gold_team:
                artifact = self._analyze_team_performance(workflow, gold_team, "gold")
                if artifact:
                    artifacts.append(artifact)

        return artifacts

    def _analyze_team_performance(self, workflow: WorkflowState, team: Team, team_role: str) -> Optional[TeamPerformanceArtifact]:
        """
        Analyze performance of a specific team.

        Args:
            workflow: The workflow state
            team: The team to analyze
            team_role: Role of the team (solver, gold, etc.)

        Returns:
            TeamPerformanceArtifact if analysis successful
        """
        # Calculate basic metrics
        total_problems = len(workflow.sub_problem_solutions)
        solved_problems = len(workflow.solved_sub_problem_ids)
        success_rate = solved_problems / total_problems if total_problems > 0 else 0.0

        # Calculate velocity (problems per hour)
        elapsed_time = time.time() - workflow.start_time
        elapsed_hours = elapsed_time / 3600

        # Prevent unrealistically high velocities from very small time windows
        min_elapsed_hours = 0.001  # 3.6 seconds minimum
        if elapsed_hours < min_elapsed_hours:
            velocity = float(solved_problems)  # Problems per second (very high but not infinite)
        else:
            velocity = solved_problems / elapsed_hours

        # Create artifact
        artifact = TeamPerformanceArtifact(
            artifact_id=f"team_perf_{team_role}_{workflow.workflow_id}_{uuid.uuid4().hex[:8]}",
            source_workflow_id=workflow.workflow_id,
            team_id=team.team_id if hasattr(team, 'team_id') else team_role,
            team_composition={"models": [m.model_id for m in team.models] if hasattr(team, 'models') else [], "role": team_role},
            velocity=velocity,
            quality_metrics={"success_rate": success_rate, "problems_solved": solved_problems},
            confidence=0.8,
        )

        # Use LLM for deeper analysis if available
        if self.llm_client:
            prompt = self.extraction_prompts["team_performance"].format(
                team_id=artifact.team_id,
                team_composition=json.dumps(artifact.team_composition),
                num_solved=solved_problems,
                total_problems=total_problems,
                quality_metrics=json.dumps(artifact.quality_metrics),
                domains=self._extract_domains_from_problem(problem),  # Extract domains from problem
                complexities=self._extract_complexities_from_problem(problem),  # Extract complexities from problem
            )
            try:
                response = self.llm_client.generate(prompt)
                extracted_data = json.loads(response)
                artifact.optimal_domains = extracted_data.get("optimal_domains", [])
                artifact.skill_gaps = extracted_data.get("skill_gaps", [])
                artifact.training_recommendations = extracted_data.get("training_recommendations", [])
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"LLM team analysis failed: {e}")

        return artifact

    # ========== Gauntlet Effectiveness Extraction ==========

    def extract_gauntlet_effectiveness(self, workflow: WorkflowState) -> List[GauntletEffectivenessArtifact]:
        """
        Extract gauntlet effectiveness insights from a workflow.

        Args:
            workflow: The workflow state

        Returns:
            List of GauntletEffectivenessArtifacts
        """
        artifacts = []

        # Analyze solver generation gauntlet
        if workflow.solver_generation_gauntlet:
            artifact = self._analyze_gauntlet_effectiveness(workflow, workflow.solver_generation_gauntlet)
            if artifact:
                artifacts.append(artifact)

        # Analyze final gold gauntlet
        if workflow.final_gold_gauntlet:
            artifact = self._analyze_gauntlet_effectiveness(workflow, workflow.final_gold_gauntlet)
            if artifact:
                artifacts.append(artifact)

        return artifacts

    def _analyze_gauntlet_effectiveness(self, workflow: WorkflowState, gauntlet: GauntletDefinition) -> Optional[GauntletEffectivenessArtifact]:
        """
        Analyze effectiveness of a specific gauntlet.

        Args:
            workflow: The workflow state
            gauntlet: The gauntlet to analyze

        Returns:
            GauntletEffectivenessArtifact if analysis successful
        """
        gauntlet_id = gauntlet.gaugment_id if hasattr(gauntlet, 'gauntlet_id') else gauntlet.name if hasattr(gauntlet, 'name') else "unknown"
        gauntlet_type = gauntlet.gaugment_type if hasattr(gauntlet, 'gauntlet_type') else "custom"

        # Calculate actual metrics from workflow data
        total_checks = len(workflow.all_verification_reports) if hasattr(workflow, 'all_verification_reports') else 1
        issues_caught = len([r for r in workflow.all_verification_reports if r and hasattr(r, 'passed') and not r.passed]) if hasattr(workflow, 'all_verification_reports') else 0
        false_positives = self._calculate_false_positives(workflow)

        catch_rate = issues_caught / total_checks if total_checks > 0 else 0.0
        false_positive_rate = false_positives / total_checks if total_checks > 0 else 0.0
        execution_time = getattr(workflow, 'execution_time', 5.0)  # Use workflow's execution time if available

        # Basic metrics (calculated from actual workflow data)
        artifact = GauntletEffectivenessArtifact(
            artifact_id=f"gauntlet_{gauntlet_id}_{workflow.workflow_id}_{uuid.uuid4().hex[:8]}",
            source_workflow_id=workflow.workflow_id,
            gauntlet_id=gauntlet_id,
            gauntlet_type=gauntlet_type,
            catch_rate=catch_rate,
            false_positive_rate=false_positive_rate,
            execution_time=execution_time,
            confidence=0.7,
        )

        # Use LLM for detailed analysis if available
        if self.llm_client:
            prompt = self.extraction_prompts["gauntlet_effectiveness"].format(
                gauntlet_id=gauntlet_id,
                gauntlet_type=gauntlet_type,
                total_checks=len(workflow.all_verification_reports),
                issues_caught=len([r for r in workflow.all_verification_reports if r and hasattr(r, 'passed') and not r.passed]),
                false_positives=self._calculate_false_positives(workflow),  # Calculate false positives
                execution_time=artifact.execution_time,
            )
            try:
                response = self.llm_client.generate(prompt)
                extracted_data = json.loads(response)
                artifact.catch_rate = extracted_data.get("catch_rate", artifact.catch_rate)
                artifact.false_positive_rate = extracted_data.get("false_positive_rate", artifact.false_positive_rate)
                artifact.rule_effectiveness = extracted_data.get("rule_effectiveness", {})
                artifact.recommended_improvements = extracted_data.get("recommended_improvements", [])
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"LLM gauntlet analysis failed: {e}")

        return artifact

    # ========== End-to-End Extraction ==========

    def extract_all_knowledge(self, workflow: WorkflowState, store: bool = True) -> Dict[str, int]:
        """
        Extract all knowledge artifacts from a workflow.

        Args:
            workflow: The workflow state
            store: Whether to store artifacts in database

        Returns:
            Dictionary with counts of extracted artifacts
        """
        counts = {
            "solution_patterns": 0,
            "team_performance": 0,
            "gauntlet_effectiveness": 0,
        }

        # Extract solution patterns
        patterns = self.extract_solution_patterns(workflow)
        counts["solution_patterns"] = len(patterns)
        if store:
            for pattern in patterns:
                self.artifact_manager.create_solution_pattern(pattern)

        # Extract team performance
        team_artifacts = self.extract_team_performance(workflow)
        counts["team_performance"] = len(team_artifacts)
        if store:
            for artifact in team_artifacts:
                self.artifact_manager.create_team_performance(artifact)

        # Extract gauntlet effectiveness
        gauntlet_artifacts = self.extract_gauntlet_effectiveness(workflow)
        counts["gauntlet_effectiveness"] = len(gauntlet_artifacts)
        if store:
            for artifact in gauntlet_artifacts:
                self.artifact_manager.create_gauntlet_effectiveness(artifact)

        return counts

    # ========== Helper Methods ==========

    def _classify_domain(self, problem_statement: str) -> str:
        """Classify the domain of a problem."""
        problem_lower = problem_statement.lower()

        domain_keywords = {
            "algorithms": ["algorithm", "sorting", "searching", "optimization"],
            "data_structures": ["array", "list", "tree", "graph", "hash"],
            "machine_learning": ["train", "model", "predict", "classify", "regression"],
            "web_development": ["api", "server", "client", "http", "database"],
            "system_design": ["scale", "distributed", "architecture", "design"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in problem_lower for kw in keywords):
                return domain

        return "general"

    def _estimate_complexity(self, problem_statement: str) -> int:
        """Estimate complexity (1-10) of a problem."""
        # Heuristic: longer problem statements tend to be more complex
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

        # Common constraint patterns
        if "time limit" in problem_lower or "timeout" in problem_lower:
            constraints.append("time_constraint")
        if "memory" in problem_lower:
            constraints.append("memory_constraint")
        if "without using" in problem_lower or "not allowed" in problem_lower:
            constraints.append("restriction")
        if "must" in problem_lower or "require" in problem_lower:
            constraints.append("requirement")

        return constraints

    def _extract_domains_from_problem(self, problem: Any) -> List[str]:
        """Extract domains from a problem object."""
        # Try to get problem statement from the problem object
        problem_statement = getattr(problem, 'problem_statement', '')
        if not problem_statement:
            problem_statement = getattr(problem, 'description', '')
        if not problem_statement:
            problem_statement = str(problem)  # fallback to string representation

        # Use existing domain classification method
        domain = self._classify_domain(problem_statement)
        return [domain]  # Return as list to match expected return type

    def _extract_complexities_from_problem(self, problem: Any) -> List[int]:
        """Extract complexities from a problem object."""
        # Try to get problem statement from the problem object
        problem_statement = getattr(problem, 'problem_statement', '')
        if not problem_statement:
            problem_statement = getattr(problem, 'description', '')
        if not problem_statement:
            problem_statement = str(problem)  # fallback to string representation

        # Use existing complexity estimation method
        complexity = self._estimate_complexity(problem_statement)
        return [complexity]  # Return as list to match expected return type

    def _calculate_false_positives(self, workflow: WorkflowState) -> int:
        """Calculate false positive rate from workflow verification reports."""
        false_positives = 0
        if hasattr(workflow, 'all_verification_reports') and workflow.all_verification_reports:
            for report in workflow.all_verification_reports:
                if report and hasattr(report, 'false_positive'):
                    if getattr(report, 'false_positive', False):
                        false_positives += 1
                # Alternative check if false_positive attribute doesn't exist
                elif hasattr(report, 'passed') and hasattr(report, 'expected_result'):
                    # If a report passed but wasn't expected to, or failed when expected to pass
                    passed = getattr(report, 'passed', False)
                    expected = getattr(report, 'expected_result', True)
                    if passed and not expected:  # This could be considered a false positive
                        false_positives += 1
        return false_positives

    # ========== OneKE Integration ==========

    async def ensure_oneke_initialized(self) -> bool:
        """
        Ensure OneKE bridge is initialized before use.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if not self.use_oneke or not self.oneke_bridge:
            return False

        if self._oneke_initialized:
            return True

        try:
            await self.oneke_bridge.initialize()
            self._oneke_initialized = True
            logger.info("OneKE bridge initialized successfully")
            return True
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to initialize OneKE bridge: {e}")
            self.oneke_bridge = None
            self._oneke_initialized = False
            return False

    async def _init_oneke_async(self) -> None:
        """Initialize OneKE bridge asynchronously."""
        if self.oneke_bridge:
            try:
                await self.oneke_bridge.initialize()
                print("OneKE bridge initialized successfully")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"Failed to initialize OneKE bridge: {e}")
                self.oneke_bridge = None
                self.use_oneke = False

    async def extract_domain_knowledge(self, workflow: WorkflowState, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract domain-specific knowledge using OneKE.

        This method fills GAP-2 (Physics Domain Knowledge) and enhances GAP-10.

        Args:
            workflow: The workflow state
            domains: List of domains to extract ('physics', 'chemistry', 'general')

        Returns:
            Dictionary of domain knowledge with extracted entities, relations, and events
        """
        if not await self.ensure_oneke_initialized():
            return {}

        if domains is None:
            # Auto-detect domains from problem statement
            domains = self._detect_domains(workflow)

        knowledge = {}

        if 'physics' in domains:
            try:
                physics_knowledge = await self.oneke_bridge.extract_physics_knowledge(workflow)
                knowledge['physics'] = physics_knowledge
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"Failed to extract physics knowledge: {e}")

        if 'chemistry' in domains:
            try:
                chemistry_knowledge = await self.oneke_bridge.extract_chemistry_knowledge(workflow)
                knowledge['chemistry'] = chemistry_knowledge
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"Failed to extract chemistry knowledge: {e}")

        # Extract general relations for all domains
        try:
            relations = await self.oneke_bridge.extract_relations(workflow)
            knowledge['relations'] = relations
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"Failed to extract relations: {e}")

        return knowledge

    def _detect_domains(self, workflow: WorkflowState) -> List[str]:
        """
        Detect relevant domains from workflow problem statement.

        Args:
            workflow: The workflow state

        Returns:
            List of detected domains
        """
        domains = []
        problem_lower = workflow.problem_statement.lower()

        # Physics keywords
        physics_keywords = [
            'quantum', 'oscillator', 'hamiltonian', 'wavefunction', 'eigen',
            'momentum', 'energy', 'particle', 'spin', 'entangle', 'schrodinger'
        ]
        if any(kw in problem_lower for kw in physics_keywords):
            domains.append('physics')

        # Chemistry keywords
        chemistry_keywords = [
            'molecule', 'atom', 'reaction', 'chemical', 'bond', 'catalyst',
            'synthesis', 'combust', 'oxid', 'reduct', 'polymer'
        ]
        if any(kw in problem_lower for kw in chemistry_keywords):
            domains.append('chemistry')

        # Default to general
        if not domains:
            domains.append('general')

        return domains

    async def extract_enhanced_solution_patterns(self, workflow: WorkflowState) -> List[SolutionPatternArtifact]:
        """
        Extract solution patterns enhanced with OneKE schema-guided extraction.

        This combines traditional extraction with OneKE for better pattern recognition.

        Args:
            workflow: The workflow state

        Returns:
            List of SolutionPatternArtifacts with enhanced information
        """
        # Extract traditional patterns
        patterns = self.extract_solution_patterns(workflow)

        # Enhance with OneKE if available
        if self.use_oneke and self.oneke_bridge:
            try:
                # Extract domain-specific patterns
                detected_domains = self._detect_domains(workflow)
                domain = detected_domains[0] if detected_domains else 'general'
                oneke_patterns = await self.oneke_bridge.extract_solution_patterns(workflow, domain)

                # Enhance existing patterns with OneKE data
                for pattern in patterns:
                    if oneke_patterns.get('patterns'):
                        pattern.domain_entities = oneke_patterns['patterns']
                    if oneke_patterns.get('techniques'):
                        pattern.enhanced_techniques = oneke_patterns['techniques']

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"Failed to enhance patterns with OneKE: {e}")

        return patterns

    async def extract_all_knowledge_enhanced(self, workflow: WorkflowState, store: bool = True) -> Dict[str, int]:
        """
        Extract all knowledge artifacts with OneKE enhancement.

        This is the enhanced version of extract_all_knowledge that includes
        domain-specific knowledge extraction.

        Args:
            workflow: The workflow state
            store: Whether to store artifacts in database

        Returns:
            Dictionary with counts of extracted artifacts
        """
        # Extract standard artifacts
        counts = self.extract_all_knowledge(workflow, store=store)

        # Extract domain knowledge if OneKE enabled
        if self.use_oneke and self.oneke_bridge:
            try:
                domain_knowledge = await self.extract_domain_knowledge(workflow)

                # Add domain knowledge counts
                for domain, knowledge in domain_knowledge.items():
                    count_key = f"{domain}_entities"
                    counts[count_key] = len(knowledge.get('entities', []) if isinstance(knowledge, dict) else [])

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"Failed to extract domain knowledge: {e}")

        return counts


# ========== Convenience Functions ==========

def extract_knowledge_from_workflow(workflow: WorkflowState, db_path: str = "./knowledge_artifacts.db", llm_client: Optional[Any] = None) -> Dict[str, int]:
    """
    Convenience function to extract all knowledge from a workflow.

    Args:
        workflow: The workflow state
        db_path: Path to artifact database
        llm_client: Optional LLM client for advanced extraction

    Returns:
        Dictionary with counts of extracted artifacts
    """
    extractor = WorkflowKnowledgeExtractor(db_path, llm_client)
    return extractor.extract_all_knowledge(workflow, store=True)

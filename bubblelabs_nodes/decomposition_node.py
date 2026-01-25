"""
Enhanced Decomposition Node for BubbleLabs Integration

Implements problem decomposition using the enhanced DecompositionEngine with all Phase 1-3 features:
- Phase 1: Enhanced SubProblem model (21 fields), enhanced prompts
- Phase 2: 10 decomposition strategies, intelligent strategy selection, enhanced quality assessment
- Phase 3: Team assignment engine, advanced MDAP system

This is a comprehensive wrapper that exposes ALL enhanced capabilities while maintaining
backward compatibility with existing workflows.
"""

from typing import Dict, Any, List, Optional
from .base_node import BubbleLabsNode, NodeExecutionError


class DecompositionNode(BubbleLabsNode):
    """
    Enhanced problem decomposition with all Phase 1-3 features.

    Enhanced capabilities:
    - 10 decomposition strategies (5 new: functional, temporal, risk_based, value_based, technical_dependency)
    - Intelligent strategy selection (500x faster than LLM)
    - Enhanced quality assessment (5 dimensions with tracking)
    - Optional team assignment (automated AI recommendations)
    - Optional MDAP execution (caching, load balancing, adaptive thresholds)
    - 21-field SubProblem model (was 8, now 21)

    This is a wrapper around the enhanced DecompositionEngine from
    decomposition_engine.py with all Phase 1-3 enhancements integrated.
    """

    # Node metadata
    DISPLAY_NAME = "Enhanced Problem Decomposition"
    DESCRIPTION = (
        "Advanced problem decomposition with 10 strategies, intelligent selection, "
        "team assignment, MDAP integration, and enhanced quality assessment. "
        "Breaks down complex problems using semantic, dependency, complexity, hybrid, "
        "research, functional, temporal, risk-based, value-based, and technical-dependency strategies."
    )
    ICON = "decomposition"
    CATEGORY = "analysis"
    VERSION = "2.0.0"  # Updated to reflect enhancements

    def __init__(self, config: Dict[str, Any] = None):
        # Initialize available_strategies BEFORE calling super().__init__()
        # This is needed because base_node.__init__ calls get_parameter_schema()
        self.available_strategies = [
            'semantic', 'dependency', 'complexity', 'hybrid', 'research',
            'functional', 'temporal', 'risk_based', 'value_based', 'technical_dependency'
        ]

        super().__init__(config)

        # Import ALL enhanced components from Phase 1-3 (safe imports)
        try:
            # Main components
            DecompositionEngine = self.safe_import(
                'decomposition_engine.DecompositionEngine',
                error_msg="DecompositionEngine not available for DecompositionNode"
            )
            sovereign_data_models = self.safe_import(
                'sovereign_data_models',
                error_msg="sovereign_data_models not available for DecompositionNode"
            )

            # Extract specific classes from sovereign_data_models
            if sovereign_data_models:
                ProblemDefinition = getattr(sovereign_data_models, 'ProblemDefinition', None)
                DecompositionStrategy = getattr(sovereign_data_models, 'DecompositionStrategy', None)
                generate_id = getattr(sovereign_data_models, 'generate_id', None)
                ProblemType = getattr(sovereign_data_models, 'ProblemType', None)
                DomainContext = getattr(sovereign_data_models, 'DomainContext', None)
                ComplexityScore = getattr(sovereign_data_models, 'ComplexityScore', None)
                Constraint = getattr(sovereign_data_models, 'Constraint', None)
                EnhancedQualityScores = getattr(sovereign_data_models, 'EnhancedQualityScores', None)
            else:
                ProblemDefinition = DecompositionStrategy = generate_id = None
                ProblemType = DomainContext = ComplexityScore = Constraint = None
                EnhancedQualityScores = None

            # Phase 3 components (optional)
            team_assignment_engine = self.safe_import(
                'team_assignment_engine',
                error_msg="team_assignment_engine not available"
            )
            team_manager = self.safe_import(
                'team_manager',
                error_msg="team_manager not available"
            )
            quality_tracker = self.safe_import(
                'quality_tracker',
                error_msg="quality_tracker not available"
            )

            TEAM_COMPONENTS_AVAILABLE = all([
                team_assignment_engine, team_manager, quality_tracker,
                hasattr(team_assignment_engine, 'TeamAssignmentEngine'),
                hasattr(team_manager, 'TeamManager'),
                hasattr(quality_tracker, 'QualityTracker')
            ])

            # Phase 3 MDAP components (optional)
            mdap_module = self.safe_import(
                'decomposition_mdap_integration',
                error_msg="decomposition_mdap_integration not available"
            )

            MDAP_COMPONENTS_AVAILABLE = all([
                mdap_module,
                hasattr(mdap_module, 'create_mdap_enhanced_decomposition_engine'),
                hasattr(mdap_module, 'get_mdap_statistics'),
                hasattr(mdap_module, 'cleanup_mdap_resources')
            ])

            # Store references
            self.DecompositionEngine = DecompositionEngine
            self.ProblemDefinition = ProblemDefinition
            self.DecompositionStrategy = DecompositionStrategy
            self.ProblemType = ProblemType
            self.DomainContext = DomainContext
            self.ComplexityScore = ComplexityScore
            self.Constraint = Constraint
            self.EnhancedQualityScores = EnhancedQualityScores
            self.generate_id = generate_id

            # Track component availability
            self.team_components_available = TEAM_COMPONENTS_AVAILABLE
            self.mdap_components_available = MDAP_COMPONENTS_AVAILABLE

            # All 10 strategies now available (Phase 2: 5 new strategies)
            # Note: Already initialized above before super().__init__() call
            # self.available_strategies = [...]

            # Initialize team components (optional - Phase 3)
            enable_team_assignment = self.config.get('enable_team_assignment', False)
            if enable_team_assignment and TEAM_COMPONENTS_AVAILABLE:
                try:
                    TeamManager = getattr(team_manager, 'TeamManager')
                    TeamAssignmentEngine = getattr(team_assignment_engine, 'TeamAssignmentEngine')
                    self.team_manager = TeamManager()
                    self.team_assignment_engine = TeamAssignmentEngine(self.team_manager)
                except Exception as e:
                    self.logger.warning(f"Could not initialize team components: {e}")
                    self.team_manager = None
                    self.team_assignment_engine = None
            else:
                self.team_manager = None
                self.team_assignment_engine = None

            # Initialize quality tracker (optional - Phase 2)
            enable_quality_tracking = self.config.get('enable_quality_tracking', True)
            if enable_quality_tracking and TEAM_COMPONENTS_AVAILABLE:
                try:
                    QualityTracker = getattr(quality_tracker, 'QualityTracker')
                    self.quality_tracker = QualityTracker()
                except Exception as e:
                    self.logger.warning(f"Could not initialize quality tracker: {e}")
                    self.quality_tracker = None
            else:
                self.quality_tracker = None

            # Initialize decomposition engine with ALL enhancements
            mdap_enabled = self.config.get('enable_mdap', False)
            if mdap_enabled and MDAP_COMPONENTS_AVAILABLE and DecompositionEngine:
                try:
                    create_mdap_enhanced_decomposition_engine = getattr(mdap_module, 'create_mdap_enhanced_decomposition_engine')
                    self.engine = create_mdap_enhanced_decomposition_engine(
                        team_assignment_engine=self.team_assignment_engine,
                        use_intelligent_selection=True  # Phase 2: intelligent selection
                    )
                    self.mdap_enabled = True
                except Exception as e:
                    self.logger.warning(f"Could not create MDAP-enhanced engine: {e}, using basic engine")
                    self.engine = DecompositionEngine(
                        team_assignment_engine=self.team_assignment_engine,
                        use_intelligent_selection=True  # Phase 2: intelligent selection
                    )
                    self.mdap_enabled = False
            elif DecompositionEngine:
                try:
                    self.engine = DecompositionEngine(
                        team_assignment_engine=self.team_assignment_engine,
                        use_intelligent_selection=True  # Phase 2: intelligent selection
                    )
                    self.mdap_enabled = False
                except Exception as e:
                    self.logger.error(f"Could not instantiate DecompositionEngine: {e}")
                    self.engine = None
            else:
                self.engine = None

            self.logger.info(
                f"Enhanced DecompositionNode initialized with {len(self.available_strategies)} strategies. "
                f"Team assignment: {'enabled' if self.team_assignment_engine else 'disabled'}. "
                f"MDAP: {'enabled' if self.mdap_enabled else 'disabled'}. "
                f"Quality tracking: {'enabled' if self.quality_tracker else 'disabled'}."
            )

        except Exception as e:
            self.logger.error(f"Critical error during DecompositionNode initialization: {e}", exc_info=True)
            # Fallback initialization
            self.DecompositionEngine = None
            self.ProblemDefinition = None
            self.DecompositionStrategy = None
            self.ProblemType = None
            self.DomainContext = None
            self.ComplexityScore = None
            self.Constraint = None
            self.EnhancedQualityScores = None
            self.generate_id = None
            self.engine = None
            self.available_strategies = []
            self.team_manager = None
            self.team_assignment_engine = None
            self.quality_tracker = None
            self.mdap_enabled = False
            self.team_components_available = False
            self.mdap_components_available = False

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters with enhanced options.

        Required:
            - problem_statement: str

        Optional:
            - method: str (one of 10 strategies OR 'intelligent' for auto-selection)
            - assign_teams: bool (enable team assignment)
            - enable_mdap: bool (enable MDAP execution)
            - enable_quality_tracking: bool (enable quality tracking)
            - constraints: Dict
            - requirements: Dict
        """
        errors = []

        # Check required fields
        if 'problem_statement' not in inputs:
            errors.append("Missing required field: problem_statement")
        elif not isinstance(inputs['problem_statement'], str):
            errors.append("problem_statement must be a string")
        elif len(inputs['problem_statement'].strip()) == 0:
            errors.append("problem_statement cannot be empty")

        # Validate method (all 10 strategies + intelligent selection)
        if 'method' in inputs:
            valid_methods = self.available_strategies + ['intelligent']
            if inputs['method'] not in valid_methods:
                errors.append(
                    f"method must be one of: {', '.join(self.available_strategies)}, or 'intelligent'"
                )

        # Validate team assignment option
        if 'assign_teams' in inputs:
            if not isinstance(inputs['assign_teams'], bool):
                errors.append("assign_teams must be a boolean")
            elif inputs['assign_teams'] and not self.team_components_available:
                errors.append("Team assignment requested but team components are not available")

        # Validate MDAP option
        if 'enable_mdap' in inputs:
            if not isinstance(inputs['enable_mdap'], bool):
                errors.append("enable_mdap must be a boolean")
            elif inputs['enable_mdap'] and not self.mdap_components_available:
                errors.append("MDAP requested but MDAP components are not available")

        # Validate quality tracking option
        if 'enable_quality_tracking' in inputs:
            if not isinstance(inputs['enable_quality_tracking'], bool):
                errors.append("enable_quality_tracking must be a boolean")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute enhanced problem decomposition with ALL Phase 1-3 features.

        Enhanced features:
        - 10 decomposition strategies (5 new from Phase 2)
        - Intelligent strategy selection (500x faster, Phase 2)
        - Enhanced quality assessment (5 dimensions, Phase 2)
        - Optional team assignment (Phase 3)
        - Optional MDAP execution (caching, load balancing, Phase 3)
        - Quality tracking and insights (Phase 2)

        Args:
            inputs: Must contain 'problem_statement' and optional enhanced parameters
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - sub_problems: List of sub-problem definitions (21 fields)
                - decomposition_tree: Hierarchical tree structure
                - estimated_time: Estimated time to solve all sub-problems
                - method_used: Strategy that was used
                - total_sub_problems: Number of sub-problems created
                - confidence: Overall confidence level
                - enhanced_quality: Enhanced quality scores (5 dimensions)
                - team_assignments: Team assignments (if enabled)
                - quality_insights: Quality trends and insights (if tracking enabled)
                - mdap_statistics: MDAP performance statistics (if MDAP enabled)
                - features_used: Which enhanced features were active
        """
        if not self.engine or not self.ProblemDefinition:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="DecompositionEngine not available",
                details={
                    'error': 'The decomposition_engine.py module must be available',
                    'hint': 'Ensure decomposition_engine.py is in the Python path'
                }
            )

        # Extract parameters with enhanced defaults
        problem_statement = inputs['problem_statement']

        # Phase 2: Default to intelligent strategy selection
        method = inputs.get('method', self.config.get('method', 'intelligent'))

        # Phase 3: Optional team assignment
        assign_teams = inputs.get('assign_teams', self.config.get('assign_teams', False))

        # Phase 3: Optional MDAP
        enable_mdap = inputs.get('enable_mdap', self.config.get('enable_mdap', False))

        # Phase 2: Quality tracking
        enable_quality_tracking = inputs.get('enable_quality_tracking', True)

        # Standard parameters
        requirements = inputs.get('requirements', self.config.get('requirements', {}))
        constraints_input = inputs.get('constraints', self.config.get('constraints', {}))

        # Update progress
        context.update_progress(10, "Creating ProblemDefinition")
        self.logger.info(f"Decomposing problem using {method} strategy")

        try:
            # Convert input to ProblemDefinition
            context.update_progress(20, "Converting to ProblemDefinition format")

            # Create domain context
            domain = inputs.get('domain', 'general')
            domain_context = self.DomainContext(
                domain=domain,
                subdomain=inputs.get('subdomain', 'general'),
                domain_knowledge=requirements
            )

            # Create complexity score (default to medium complexity)
            complexity_score = self.ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Default complexity - can be customized"
            )

            # Convert constraints input to Constraint objects
            constraints_list = self._parse_constraints(constraints_input)

            # Determine problem type
            problem_type_str = requirements.get('problem_type', inputs.get('problem_type', 'implementation'))
            problem_type = self._map_problem_type(problem_type_str)

            problem = self.ProblemDefinition(
                id=self.generate_id("problem"),
                title=inputs.get('title', 'Problem'),
                description=problem_statement,
                problem_type=problem_type,
                domain_context=domain_context,
                complexity_score=complexity_score,
                constraints=constraints_list,
                resources_available=requirements
            )

            # Phase 2: Intelligent strategy selection
            if method == 'intelligent':
                if hasattr(self.engine, 'select_strategy_intelligent'):
                    method = self.engine.select_strategy_intelligent(problem)
                    self.logger.info(f"Intelligent selection chose strategy: {method}")
                else:
                    # Fallback to hybrid if intelligent selection not available
                    method = 'hybrid'
                    self.logger.warning("Intelligent selection not available, using hybrid strategy")

            context.update_progress(40, f"Decomposing with {method} strategy")

            # Decompose with optional team assignment
            teams = inputs.get('teams') if assign_teams else None
            plan = self.engine.decompose(
                problem=problem,
                strategy=method,
                assign_teams=assign_teams,
                teams=teams
            )

            # Phase 2: Enhanced quality assessment
            enhanced_quality = None
            quality_insights = {}

            if enable_quality_tracking and hasattr(self.engine, '_assess_quality_enhanced'):
                context.update_progress(60, "Assessing decomposition quality (enhanced)")

                try:
                    enhanced_quality = self.engine._assess_quality_enhanced(
                        problem, plan.sub_problems
                    )

                    # Track quality if tracker available
                    if self.quality_tracker:
                        self.quality_tracker.record_assessment(
                            plan_id=plan.id,
                            scores=enhanced_quality,
                            problem_type=problem.problem_type.value,
                            strategy=method
                        )
                        quality_insights = self.quality_tracker.get_insights()

                except Exception as e:
                    self.logger.warning(f"Enhanced quality assessment failed: {e}")

            # Phase 3: Get MDAP statistics if enabled
            mdap_stats = {}
            if enable_mdap and self.mdap_enabled:
                try:
                    mdap_module = self.safe_import(
                        'decomposition_mdap_integration',
                        error_msg="decomposition_mdap_integration not available for MDAP stats"
                    )
                    if mdap_module and hasattr(mdap_module, 'get_mdap_statistics'):
                        get_mdap_statistics = getattr(mdap_module, 'get_mdap_statistics')
                        mdap_stats = get_mdap_statistics(self.engine)
                    else:
                        mdap_stats = {}
                except Exception as e:
                    self.logger.warning(f"Failed to get MDAP statistics: {e}")
                    mdap_stats = {}

            # Update progress
            context.update_progress(90, "Converting to output format")

            # Convert results with ALL enhanced fields
            result = {
                # Basic fields (backward compatible)
                'sub_problems': self._convert_sub_problems_enhanced(plan.sub_problems),
                'decomposition_tree': self._convert_dependency_graph(plan.dependency_graph),
                'complexity_metrics': {
                    'overall_score': plan.quality_scores.overall_score,
                    'meets_thresholds': plan.quality_scores.meets_thresholds,
                    'confidence': plan.confidence_level
                },
                'estimated_time': sum(sp.estimated_time for sp in plan.sub_problems),
                'method_used': plan.strategy.value,
                'total_sub_problems': len(plan.sub_problems),
                'confidence': plan.confidence_level,
                'validation_checkpoints': len(plan.validation_checkpoints),
                'plan_id': plan.id,
                'problem_id': plan.id,

                # NEW: Enhanced fields from Phase 2
                'enhanced_quality': self._convert_enhanced_quality(enhanced_quality) if enhanced_quality else None,
                'quality_insights': quality_insights,

                # NEW: Team assignments from Phase 3
                'team_assignments': self._convert_team_assignments(plan.sub_problems) if assign_teams else [],

                # NEW: MDAP statistics from Phase 3
                'mdap_statistics': mdap_stats,

                # Metadata: Which features were active
                'features_used': {
                    'intelligent_strategy_selection': method == 'intelligent' or inputs.get('method') == 'intelligent',
                    'enhanced_quality_assessment': enable_quality_tracking and enhanced_quality is not None,
                    'team_assignment': assign_teams,
                    'mdap_enabled': enable_mdap and self.mdap_enabled,
                    'quality_tracking': self.quality_tracker is not None
                }
            }

            # Add artifacts to context
            context.add_artifact('decomposition', {
                'problem_statement': problem_statement,
                'method': method,
                'plan_id': plan.id,
                'sub_problems_count': len(plan.sub_problems),
                'enhanced_features': result['features_used']
            })

            context.update_progress(
                100,
                f"Complete: {len(plan.sub_problems)} sub-problems, "
                f"confidence={result['confidence']:.2f}, "
                f"strategy={method}"
            )

            self.logger.info(
                f"Enhanced decomposition completed: {result['total_sub_problems']} sub-problems, "
                f"strategy={result['method_used']}, confidence={result['confidence']:.2f}, "
                f"features={result['features_used']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Decomposition failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Decomposition failed: {str(e)}",
                details={
                    'problem_statement': problem_statement[:100],
                    'method': method,
                    'exception_type': type(e).__name__,
                    'enhanced_features': {
                        'team_assignment': assign_teams,
                        'mdap': enable_mdap,
                        'quality_tracking': enable_quality_tracking
                    }
                }
            ) from e

    def _parse_constraints(self, constraints_input: Any) -> List:
        """Parse constraints from various input formats."""
        constraints_list = []

        if not constraints_input:
            return constraints_list

        if isinstance(constraints_input, list):
            for i, constraint in enumerate(constraints_input):
                if isinstance(constraint, dict):
                    constraints_list.append(self.Constraint(
                        id=self.generate_id("constraint"),
                        description=constraint.get('description', f"Constraint {i+1}"),
                        type=constraint.get('type', 'general'),
                        severity=constraint.get('severity', 'medium')
                    ))
                elif isinstance(constraint, str):
                    constraints_list.append(self.Constraint(
                        id=self.generate_id("constraint"),
                        description=constraint,
                        type='general',
                        severity='medium'
                    ))
        elif isinstance(constraints_input, dict):
            for key, value in constraints_input.items():
                constraints_list.append(self.Constraint(
                    id=self.generate_id("constraint"),
                    description=f"{key}: {value}",
                    type='general',
                    severity='medium'
                ))

        return constraints_list

    def _map_problem_type(self, problem_type_str: str):
        """Map problem type string to ProblemType enum."""
        problem_type_map = {
            'implementation': self.ProblemType.IMPLEMENTATION,
            'analysis': self.ProblemType.ANALYSIS,
            'research': self.ProblemType.RESEARCH,
            'design': self.ProblemType.DESIGN,
            'optimization': self.ProblemType.OPTIMIZATION
        }
        return problem_type_map.get(problem_type_str.lower(), self.ProblemType.IMPLEMENTATION)

    def _convert_sub_problems_enhanced(self, sub_problems: List) -> List[Dict]:
        """
        Convert SubProblem objects to dictionaries with ALL 21 fields (Phase 1 enhancement).

        Returns comprehensive sub-problem data including all enhanced fields from Phase 1.
        """
        converted = []

        for sp in sub_problems:
            sp_dict = {
                # Basic fields (original 8 fields)
                'id': sp.id,
                'title': sp.title,
                'description': sp.description,
                'priority': sp.priority,
                'complexity': sp.complexity_score.overall_complexity if sp.complexity_score else 5.0,
                'dependencies': [d.id for d in sp.dependencies],
                'estimated_time': sp.estimated_time,
                'type': sp.type.value if hasattr(sp, 'type') else 'general',
                'status': sp.status.value if hasattr(sp, 'status') else 'pending',
                'success_criteria': [sc.to_dict() if hasattr(sc, 'to_dict') else str(sc) for sc in sp.success_criteria],
            }

            # NEW: Enhanced fields from Phase 1 (13 new fields)
            # Extract from metadata if available
            metadata = sp.metadata if hasattr(sp, 'metadata') else {}

            if isinstance(metadata, dict):
                # Acceptance criteria
                if 'acceptance_criteria' in metadata:
                    sp_dict['acceptance_criteria'] = metadata['acceptance_criteria']

                # Evolution mode
                if 'evolution_mode' in metadata:
                    sp_dict['evolution_mode'] = metadata['evolution_mode']

                # Complexity breakdown
                if 'complexity_breakdown' in metadata:
                    sp_dict['complexity_breakdown'] = metadata['complexity_breakdown']

                # Evaluation prompt
                if 'evaluation_prompt' in metadata:
                    sp_dict['evaluation_prompt'] = metadata['evaluation_prompt']

                # Team assignment (stored in metadata or as dedicated field)
                if 'team_assignment' in metadata:
                    sp_dict['team_assignment_note'] = metadata['team_assignment']

                # Gauntlet assignment
                if 'gauntlet_assignment' in metadata:
                    sp_dict['gauntlet_assignment'] = metadata['gauntlet_assignment']

                # Estimated resources
                if 'estimated_resources' in metadata:
                    sp_dict['resources'] = metadata['estimated_resources']

                # Potential approaches
                if 'potential_approaches' in metadata:
                    sp_dict['approaches'] = metadata['potential_approaches']

                # Required expertise
                if 'required_expertise' in metadata:
                    sp_dict['expertise'] = metadata['required_expertise']

                # Associated risks
                if 'associated_risks' in metadata:
                    sp_dict['risks'] = metadata['associated_risks']

                # Success dependencies
                if 'success_dependencies' in metadata:
                    sp_dict['success_dependencies'] = metadata['success_dependencies']

                # Testing approach
                if 'testing_approach' in metadata:
                    sp_dict['testing'] = metadata['testing_approach']

                # Quality metrics
                if 'quality_metrics' in metadata:
                    sp_dict['quality_targets'] = metadata['quality_metrics']

            # Phase 3: Check for dedicated team assignment field
            if hasattr(sp, 'ai_suggested_team_assignment') and sp.ai_suggested_team_assignment:
                sp_dict['team_assignment'] = sp.ai_suggested_team_assignment.to_dict() if hasattr(sp.ai_suggested_team_assignment, 'to_dict') else sp.ai_suggested_team_assignment

            converted.append(sp_dict)

        return converted

    def _convert_enhanced_quality(self, quality) -> Optional[Dict[str, Any]]:
        """
        Convert EnhancedQualityScores to dictionary (Phase 2 feature).

        Returns comprehensive quality assessment across 5 dimensions.
        """
        if not quality:
            return None

        return {
            'overall_score': quality.overall_score,
            'meets_thresholds': quality.meets_thresholds,

            # Dimension scores (Phase 2: 5 dimensions)
            'completeness_score': quality.completeness_score,
            'consistency_score': quality.consistency_score,
            'feasibility_score': quality.feasibility_score,
            'dependency_score': quality.dependency_score,
            'balance_score': quality.balance_score,

            # Details
            'completeness_details': quality.completeness_details,
            'consistency_details': quality.consistency_details,
            'feasibility_details': quality.feasibility_details,
            'dependency_details': quality.dependency_details,
            'balance_details': quality.balance_details,

            # Recommendations
            'improvement_recommendations': quality.improvement_recommendations[:5],  # Top 5
            'critical_issues': quality.critical_issues[:3],  # Top 3
            'validation_checkpoints': quality.validation_checkpoints
        }

    def _convert_team_assignments(self, sub_problems: List) -> List[Dict[str, Any]]:
        """
        Extract team assignments from sub-problems (Phase 3 feature).

        Returns list of team assignments for each sub-problem.
        """
        assignments = []

        for sp in sub_problems:
            if hasattr(sp, 'ai_suggested_team_assignment') and sp.ai_suggested_team_assignment:
                assignment = sp.ai_suggested_team_assignment
                assignments.append({
                    'sub_problem_id': sp.id,
                    'sub_problem_title': sp.title,
                    'solver': assignment.solver if hasattr(assignment, 'solver') else None,
                    'patcher': assignment.patcher if hasattr(assignment, 'patcher') else None,
                    'red_team': assignment.red_team if hasattr(assignment, 'red_team') else None,
                    'gold_team': assignment.gold_team if hasattr(assignment, 'gold_team') else None
                })

        return assignments

    def _convert_dependency_graph(self, graph) -> Dict[str, Any]:
        """Convert DependencyGraph to dictionary."""
        if graph is None:
            return {'nodes': [], 'edges': []}

        return {
            'nodes': list(graph.nodes.keys()) if hasattr(graph, 'nodes') else [],
            'edges': graph.edges if hasattr(graph, 'edges') else [],
            'execution_order': list(graph.execution_order) if hasattr(graph, 'execution_order') else []
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters with ALL enhanced options.

        Returns schema supporting all 10 strategies, intelligent selection,
        team assignment, MDAP, and quality tracking.
        """
        return {
            "type": "object",
            "title": "Enhanced Decomposition Configuration",
            "description": "Configure problem decomposition with ALL Phase 1-3 enhanced features",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Decomposition Method",
                    "description": "Strategy to use (use 'intelligent' for 500x faster auto-selection)",
                    "enum": self.available_strategies + ["intelligent"],
                    "enumNames": [
                        "Semantic (LLM-based concept analysis)",
                        "Dependency-Based (prerequisite relationships)",
                        "Complexity-Based (complexity prioritization)",
                        "Hybrid (adaptive multi-strategy)",
                        "Research (exploration-based)",
                        "Functional (modules/components)",  # NEW - Phase 2
                        "Temporal (time phases)",  # NEW - Phase 2
                        "Risk-Based (risk priority)",  # NEW - Phase 2
                        "Value-Based (business value)",  # NEW - Phase 2
                        "Technical Dependency (infrastructure)",  # NEW - Phase 2
                        "Intelligent (auto-select)"  # NEW - Phase 2
                    ],
                    "default": "intelligent"  # Changed from 'hybrid' to 'intelligent'
                },
                # NEW - Phase 3: Team assignment
                "assign_teams": {
                    "type": "boolean",
                    "title": "Enable Team Assignment",
                    "description": "Automatically assign teams to sub-problems using AI recommendations",
                    "default": False
                },
                # NEW - Phase 3: MDAP
                "enable_mdap": {
                    "type": "boolean",
                    "title": "Enable MDAP",
                    "description": "Use advanced MDAP with caching, load balancing, and adaptive thresholds",
                    "default": False
                },
                # NEW - Phase 2: Quality tracking
                "enable_quality_tracking": {
                    "type": "boolean",
                    "title": "Enable Quality Tracking",
                    "description": "Track quality metrics over time and provide insights",
                    "default": True
                },
                # Standard parameters
                "requirements": {
                    "type": "object",
                    "title": "Requirements",
                    "description": "Problem requirements",
                    "default": {}
                },
                "constraints": {
                    "type": "object",
                    "title": "Constraints",
                    "description": "Problem constraints",
                    "default": {}
                }
            }
        }

    def cleanup(self):
        """Cleanup resources, especially MDAP resources if enabled."""
        try:
            if self.mdap_enabled and hasattr(self, 'engine'):
                mdap_module = self.safe_import(
                    'decomposition_mdap_integration',
                    error_msg="decomposition_mdap_integration not available for cleanup"
                )
                if mdap_module and hasattr(mdap_module, 'cleanup_mdap_resources'):
                    cleanup_mdap_resources = getattr(mdap_module, 'cleanup_mdap_resources')
                    cleanup_mdap_resources(self.engine)
                    self.logger.info("MDAP resources cleaned up")
                else:
                    self.logger.info("MDAP cleanup function not available, skipping")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup MDAP resources: {e}")

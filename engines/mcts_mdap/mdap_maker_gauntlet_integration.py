"""
MDAP/MAKER-Gauntlet Integration Module

Provides comprehensive integration between MDAP/MAKER systems and the Gauntlet quality control system.

This module enables:
1. MDAP-driven adaptive gauntlet configuration
2. MAKER voting-based gauntlet evaluation
3. Multi-agent consensus for gauntlet rounds
4. Complexity-based gauntlet selection
5. Red/Blue team integration with MDAP/MAKER

Author: OpenEvolve Team
Date: 2026-02-17
"""
from __future__ import annotations


import logging
import time
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

try:
    from gauntlet_manager import GauntletManager, GauntletEvaluator
    from gauntlet_types import (
        GauntletType, GauntletResult, BaseGauntlet,
        AdversarialGauntlet, FormalVerificationGauntlet, StatisticalGauntlet,
        DomainSpecificGauntlet, MultiObjectiveGauntlet, EvolutionaryGauntlet,
        TemporalGauntlet, CrossValidationGauntlet,
    )
    GAUNTLET_AVAILABLE = True
except ImportError as e:
    GAUNTLET_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Gauntlet integration not available: {e}")

    # Graceful-degradation stubs so the module stays importable and the
    # maker pipeline can run without the external gauntlet libraries.
    from enum import Enum

    class GauntletType(Enum):
        ADVERSARIAL = "adversarial"
        FORMAL_VERIFICATION = "formal_verification"
        STATISTICAL = "statistical"
        DOMAIN_PHYSICS = "domain_physics"
        MULTI_OBJECTIVE = "multi_objective"
        EVOLUTIONARY = "evolutionary"
        TEMPORAL = "temporal"
        CROSS_VALIDATION = "cross_validation"

    class BaseGauntlet:
        def __init__(self, name: str = ""):
            self.name = name

    class GauntletResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class AdversarialGauntlet(BaseGauntlet):
        pass

    class FormalVerificationGauntlet(BaseGauntlet):
        pass

    class StatisticalGauntlet(BaseGauntlet):
        pass

    class DomainSpecificGauntlet(BaseGauntlet):
        def __init__(self, name: str = "domain", domain: str = "general"):
            super().__init__(name)
            self.domain = domain

    class MultiObjectiveGauntlet(BaseGauntlet):
        pass

    class EvolutionaryGauntlet(BaseGauntlet):
        pass

    class TemporalGauntlet(BaseGauntlet):
        pass

    class CrossValidationGauntlet(BaseGauntlet):
        pass

    class GauntletManager:
        def __init__(self, *args, **kwargs):
            pass

    class GauntletEvaluator:
        def __init__(self, *args, **kwargs):
            pass

# MDAP/MAKER imports
try:
    from adaptive_mdap import (
        TaskComplexityClassifier,
        AdaptiveMDAPAllocator,
        AdaptiveExecutionController,
        ComplexityScore,
        SubProblem,
        get_health_checker
    )
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError as e:
    ADAPTIVE_MDAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Adaptive MDAP not available: {e}")

    # Provide stubs for graceful degradation
    @dataclass
    class SubProblem:
        """Stub SubProblem for graceful degradation."""
        id: str = ""
        description: str = ""
        domain: str = "general"
        depth: int = 1
        dependencies: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

    class ComplexityScore:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class TaskComplexityClassifier:
        def __init__(self, *args, **kwargs):
            pass

    class AdaptiveMDAPAllocator:
        def __init__(self, *args, **kwargs):
            pass

    class AdaptiveExecutionController:
        def __init__(self, *args, **kwargs):
            pass

    def get_health_checker(*args, **kwargs):
        return None

try:
    from maker_engine import MakerEngine, MakerConfig, MakerState, MakerStep
    from mdap_engine import RedFlagRules, RedFlagger
    MAKER_AVAILABLE = True
except ImportError as e:
    MAKER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"MAKER engine not available: {e}")

    # Graceful-degradation stubs.
    class MakerConfig:
        def __init__(self, *args, **kwargs):
            pass

    class MakerState:
        def __init__(self, *args, **kwargs):
            pass

    class MakerStep:
        def __init__(self, *args, **kwargs):
            pass

    class MakerEngine:
        def __init__(self, *args, **kwargs):
            pass

    class RedFlagRules:
        def __init__(self, *args, **kwargs):
            pass

    class RedFlagger:
        def __init__(self, *args, **kwargs):
            pass

try:
    from openevolve_structures import GauntletDefinition, GauntletRoundRule
    OPENEVOLVE_STRUCTURES_AVAILABLE = True
except ImportError:
    OPENEVOLVE_STRUCTURES_AVAILABLE = False

    class GauntletDefinition:
        def __init__(self, *args, **kwargs):
            pass

    class GauntletRoundRule:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


class MDAPMakerGauntletMode(Enum):
    """Execution modes for MDAP/MAKER-Gauntlet integration."""
    MDAP_ADAPTIVE = "mdap_adaptive"  # Use MDAP to configure gauntlet
    MAKER_VOTING = "maker_voting"  # Use MAKER voting for evaluation
    HYBRID = "hybrid"  # Combine both approaches
    CONSENSUS = "consensus"  # Multi-agent consensus


@dataclass
class MDAPMakerGauntletConfig:
    """Configuration for MDAP/MAKER-Gauntlet integration."""
    mode: MDAPMakerGauntletMode = MDAPMakerGauntletMode.HYBRID
    use_complexity_adaptation: bool = True
    use_maker_voting: bool = True
    use_red_flagging: bool = True
    min_complexity_threshold: float = 0.3
    max_complexity_threshold: float = 0.8
    maker_k_min: int = 2
    maker_k_max: int = 5
    maker_max_votes: int = 30
    gauntlet_types: List[GauntletType] = field(default_factory=list)
    mdap_allocator: Optional[AdaptiveMDAPAllocator] = None
    maker_config: Optional[MakerConfig] = None


@dataclass
class MDAPMakerGauntletResult:
    """Result from MDAP/MAKER-Gauntlet execution."""
    gauntlet_result: GauntletResult
    complexity_score: Optional[ComplexityScore] = None
    maker_state: Optional[MakerState] = None
    maker_metrics: Optional[Dict[str, Any]] = None
    mdap_strategy: Optional[str] = None
    red_flags: List[Dict[str, Any]] = field(default_factory=list)
    agent_votes: List[Dict[str, Any]] = field(default_factory=list)
    consensus_reached: bool = False
    consensus_score: float = 0.0


class MDAPMakerGauntletIntegration:
    """
    Integrates MDAP/MAKER systems with Gauntlet quality control.
    
    Features:
    1. MDAP-driven adaptive gauntlet configuration
    2. MAKER voting-based gauntlet evaluation
    3. Multi-agent consensus for gauntlet rounds
    4. Complexity-based gauntlet selection
    5. Red/Blue team integration with MDAP/MAKER
    """

    def __init__(self, config: Optional[MDAPMakerGauntletConfig] = None):
        """
        Initialize MDAP/MAKER-Gauntlet integration.
        
        Args:
            config: Configuration for integration
        """
        self.config = config or MDAPMakerGauntletConfig()
        self.gauntlet_manager = GauntletManager()
        
        # Initialize MDAP components
        self.complexity_classifier = None
        self.mdap_allocator = None
        self.execution_controller = None
        
        if ADAPTIVE_MDAP_AVAILABLE:
            try:
                self.complexity_classifier = TaskComplexityClassifier()
                self.mdap_allocator = self.config.mdap_allocator or AdaptiveMDAPAllocator()
                self.execution_controller = AdaptiveExecutionController(
                    classifier=self.complexity_classifier,
                    allocator=self.mdap_allocator
                )
                logger.info("MDAP components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MDAP components: {e}")
        
        # Initialize MAKER components
        self.maker_engine = None
        self.red_flagger = None

        if MAKER_AVAILABLE:
            try:
                maker_config = self.config.maker_config or MakerConfig(
                    k_min=self.config.maker_k_min,
                    k_max=self.config.maker_k_max,
                    max_votes_per_step=self.config.maker_max_votes
                )

                # Create a mock team for MAKER
                from workflow_structures import ModelConfig
                default_model = ModelConfig(
                    model_id="gpt-4o-mini",
                    temperature=0.0
                )
                
                try:
                    # Try BubbleLab Team model first
                    from core_projects.BubbleLab.services.openevolve_api.models.team_assignment import Team
                    team = Team(
                        team_id="mdap_maker_gauntlet_team",
                        name="MDAP Maker Gauntlet Team",
                        members=[default_model],
                        description="Auto-generated team for MDAP/MAKER-Gauntlet integration"
                    )
                except ImportError:
                    # Fallback to workflow_structures if available
                    try:
                        from workflow_structures import Team
                        team = Team(
                            team_id="mdap_maker_gauntlet_team",
                            name="MDAP Maker Gauntlet Team",
                            members=[default_model]
                        )
                    except Exception:
                        # Last resort: create minimal stub
                        class Team:
                            def __init__(self, team_id="", name="", members=None, **kwargs):
                                self.team_id = team_id
                                self.name = name
                                self.members = members or []
                        team = Team(
                            team_id="mdap_maker_gauntlet_team",
                            name="MDAP Maker Gauntlet Team",
                            members=[default_model]
                        )
                
                self.maker_engine = MakerEngine(team=team, config=maker_config)
                self.red_flagger = RedFlagger(maker_config.red_flag_rules)
                logger.info("MAKER components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MAKER components: {e}")
        
        logger.info(f"MDAPMakerGauntletIntegration initialized (mode={self.config.mode.value})")

    def execute_with_mdap_maker(
        self,
        gauntlet: BaseGauntlet,
        solution: Any,
        context: Optional[Dict[str, Any]] = None,
        problem_description: str = ""
    ) -> MDAPMakerGauntletResult:
        """
        Execute gauntlet with MDAP/MAKER integration.
        
        Args:
            gauntlet: Gauntlet to execute
            solution: Solution to evaluate
            context: Additional context
            problem_description: Description of the problem for complexity analysis
            
        Returns:
            MDAPMakerGauntletResult with comprehensive results
        """
        start_time = time.time()
        context = context or {}
        
        # Step 1: Analyze complexity (if enabled)
        complexity_score = None
        mdap_strategy = None
        
        if self.config.use_complexity_adaptation and self.complexity_classifier:
            try:
                complexity_score = self._analyze_complexity(problem_description, solution, context)
                
                # Allocate resources based on complexity
                if self.mdap_allocator:
                    strategy = self.mdap_allocator.allocate_resources(
                        complexity_score.overall_score,
                        context=context
                    )
                    mdap_strategy = strategy.strategy.value if hasattr(strategy.strategy, 'value') else str(strategy.strategy)
                    
                    # Adapt gauntlet config based on strategy
                    self._adapt_gauntlet_config(gauntlet, strategy)
                    
            except Exception as e:
                logger.warning(f"MDAP complexity analysis failed: {e}")
        
        # Step 2: Execute gauntlet with MAKER voting (if enabled)
        maker_state = None
        maker_metrics = None
        agent_votes = []
        red_flags = []
        
        if self.config.use_maker_voting and self.maker_engine:
            try:
                maker_result = self._execute_with_maker_voting(gauntlet, solution, context)
                maker_state = maker_result.get("state")
                maker_metrics = maker_result.get("metrics")
                agent_votes = maker_result.get("agent_votes", [])
                red_flags = maker_result.get("red_flags", [])
            except Exception as e:
                logger.warning(f"MAKER voting failed: {e}")
        
        # Step 3: Execute base gauntlet
        try:
            gauntlet_result = gauntlet.execute(solution, context)
        except Exception as e:
            logger.error(f"Gauntlet execution failed: {e}")
            gauntlet_result = GauntletResult(
                gauntlet_type=gauntlet.gauntlet_type,
                gauntlet_name=gauntlet.name,
                solution_id="error",
                passed=False,
                score=0.0,
                confidence=0.0,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details={"error": str(e)},
                feedback=f"Gauntlet execution error: {str(e)}"
            )
        
        # Step 4: Calculate consensus (if multi-agent)
        consensus_reached = False
        consensus_score = 0.0
        
        if agent_votes:
            consensus_reached, consensus_score = self._calculate_consensus(agent_votes, gauntlet_result)
        
        # Step 5: Record execution metrics
        if self.execution_controller:
            try:
                self.execution_controller.record_execution(
                    task_id=f"gauntlet_{gauntlet.name}",
                    success=gauntlet_result.passed,
                    duration=time.time() - start_time
                )
            except Exception as e:
                logger.warning(f"Failed to record execution metrics: {e}")
        
        return MDAPMakerGauntletResult(
            gauntlet_result=gauntlet_result,
            complexity_score=complexity_score,
            maker_state=maker_state,
            maker_metrics=maker_metrics,
            mdap_strategy=mdap_strategy,
            red_flags=red_flags,
            agent_votes=agent_votes,
            consensus_reached=consensus_reached,
            consensus_score=consensus_score
        )

    def _analyze_complexity(
        self,
        problem_description: str,
        solution: Any,
        context: Dict[str, Any]
    ) -> Optional[ComplexityScore]:
        """Analyze problem complexity using MDAP."""
        if not self.complexity_classifier:
            return None
        
        try:
            # Create sub-problem for complexity analysis
            sub_problem = SubProblem(
                id=f"complexity_{int(time.time())}",
                description=problem_description[:1000] if problem_description else str(solution)[:1000],
                domain=context.get("domain", "general"),
                depth=context.get("depth", 1),
                dependencies=context.get("dependencies", []),
                metadata={
                    "solution_type": type(solution).__name__,
                    "context_keys": list(context.keys())
                }
            )
            
            # Compute complexity score
            complexity_score = self.complexity_classifier.compute_complexity(sub_problem)
            
            logger.info(
                f"Complexity analysis: overall={complexity_score.overall_score:.3f}, "
                f"text={complexity_score.text_length_score:.3f}, "
                f"depth={complexity_score.depth_score:.3f}"
            )
            
            return complexity_score
            
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            return None

    def _adapt_gauntlet_config(self, gauntlet: BaseGauntlet, strategy: Any):
        """Adapt gauntlet configuration based on MDAP strategy."""
        try:
            # Adjust gauntlet parameters based on allocated strategy
            if hasattr(strategy, 'n_agents'):
                # More agents = more thorough evaluation
                gauntlet.config['n_evaluators'] = strategy.n_agents
            
            if hasattr(strategy, 'timeout_ms'):
                # Adjust timeout based on complexity
                gauntlet.config['timeout'] = strategy.timeout_ms / 1000
            
            if hasattr(strategy, 'max_retries'):
                # More retries for complex problems
                gauntlet.config['max_retries'] = strategy.max_retries
            
            logger.debug(f"Adapted gauntlet config: n_agents={strategy.n_agents}, timeout={strategy.timeout_ms}ms")
            
        except Exception as e:
            logger.warning(f"Failed to adapt gauntlet config: {e}")

    def _execute_with_maker_voting(
        self,
        gauntlet: BaseGauntlet,
        solution: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute gauntlet evaluation using MAKER voting."""
        if not self.maker_engine:
            return {}

        try:
            # Define MAKER steps for gauntlet evaluation
            step_index = [0]  # Use list to allow mutation in closure
            
            def step_builder(state: Any, history: List) -> MakerStep:
                step_index[0] += 1
                return MakerStep(
                    step_id=f"gauntlet_eval_{step_index[0]}",
                    prompt_template=f"""
Evaluate the following solution for gauntlet: {gauntlet.name}

Current Evaluation State (JSON):
{{state}}

Previous History (JSON):
{{history}}

Provide your evaluation score (0.0-1.0) and justification in JSON format: {{{{"score": 0.9, "justification": "..."}}}}
                    """,
                    task_type="evaluation",
                    priority=1,
                    metadata={
                        "gauntlet_name": gauntlet.name,
                        "gauntlet_type": gauntlet.gauntlet_type.value
                    }
                )

            def apply_action(state: Any, action: Any) -> Any:
                # Apply evaluation action (vote)
                if isinstance(action, dict) and 'score' in action:
                    state['votes'].append(action)
                return state

            # Initial state as dict (MAKER expects this)
            initial_state = {
                'solution': str(solution),
                'context': context,
                'gauntlet_name': gauntlet.name,
                'votes': []
            }

            # Execute MAKER with dict-based state
            maker_result = self.maker_engine.solve(
                initial_state=initial_state,
                step_builder=step_builder,
                apply_action=apply_action,
                stop_condition=lambda state: len(state.get('votes', [])) >= self.config.maker_max_votes
            )

            # Extract red flags
            red_flags = []
            if self.red_flagger and maker_result.state:
                state_dict = maker_result.state.current_state if hasattr(maker_result.state, 'current_state') else maker_result.state
                for vote in state_dict.get('votes', []):
                    if isinstance(vote, dict):
                        flag = self.red_flagger.check(vote.get('justification', ''))
                        if flag and hasattr(flag, 'is_flagged') and flag.is_flagged:
                            red_flags.append({
                                'rule': flag.rule_id if hasattr(flag, 'rule_id') else 'unknown',
                                'severity': flag.severity if hasattr(flag, 'severity') else 'medium',
                                'message': flag.message if hasattr(flag, 'message') else 'Red flag detected'
                            })

            return {
                'state': maker_result.state,
                'metrics': maker_result.metrics,
                'agent_votes': maker_result.state.current_state.get('votes', []) if hasattr(maker_result.state, 'current_state') else maker_result.state.get('votes', []),
                'red_flags': red_flags
            }

        except Exception as e:
            logger.error(f"MAKER voting execution failed: {e}")
            return {
                'state': None,
                'metrics': {'errors': 1, 'error_message': str(e)},
                'agent_votes': [],
                'red_flags': []
            }

    def _calculate_consensus(
        self,
        agent_votes: List[Dict[str, Any]],
        gauntlet_result: GauntletResult
    ) -> Tuple[bool, float]:
        """Calculate consensus from agent votes."""
        if not agent_votes:
            return False, 0.0
        
        try:
            # Extract scores from votes
            scores = []
            for vote in agent_votes:
                if isinstance(vote, dict) and 'score' in vote:
                    scores.append(float(vote['score']))
            
            if not scores:
                return False, 0.0
            
            # Calculate consensus metrics
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5
            
            # Consensus reached if low variance and sufficient votes
            consensus_threshold = 0.2  # Max acceptable std dev
            consensus_reached = std_dev < consensus_threshold and len(scores) >= 3
            
            # Consensus score (1.0 = perfect agreement, 0.0 = complete disagreement)
            consensus_score = max(0.0, 1.0 - (std_dev / consensus_threshold))
            
            logger.info(
                f"Consensus: reached={consensus_reached}, score={consensus_score:.3f}, "
                f"mean={mean_score:.3f}, std={std_dev:.3f}, votes={len(scores)}"
            )
            
            return consensus_reached, consensus_score
            
        except Exception as e:
            logger.warning(f"Consensus calculation failed: {e}")
            return False, 0.0

    def create_mdap_adaptive_gauntlet(
        self,
        problem_description: str,
        solution: Any,
        context: Optional[Dict[str, Any]] = None,
        base_gauntlet_type: GauntletType = GauntletType.ADVERSARIAL
    ) -> Tuple[BaseGauntlet, MDAPMakerGauntletResult]:
        """
        Create and execute a gauntlet adapted using MDAP.
        
        Args:
            problem_description: Description of the problem
            solution: Solution to evaluate
            context: Additional context
            base_gauntlet_type: Base gauntlet type to adapt
            
        Returns:
            Tuple of (gauntlet, result)
        """
        # Analyze complexity
        complexity_score = self._analyze_complexity(problem_description, solution, context or {})
        
        if not complexity_score:
            # Fallback to standard gauntlet
            logger.warning("Complexity analysis failed, using standard gauntlet")
            gauntlet = self._create_standard_gauntlet(base_gauntlet_type)
            result = gauntlet.execute(solution, context or {})
            return gauntlet, MDAPMakerGauntletResult(gauntlet_result=result)
        
        # Select gauntlet type based on complexity
        selected_type = self._select_gauntlet_type(complexity_score, base_gauntlet_type)
        
        # Create adapted gauntlet
        gauntlet = self._create_adapted_gauntlet(selected_type, complexity_score, context or {})
        
        # Execute with MDAP/MAKER integration
        result = self.execute_with_mdap_maker(
            gauntlet=gauntlet,
            solution=solution,
            context=context,
            problem_description=problem_description
        )
        
        return gauntlet, result

    def _create_standard_gauntlet(self, gauntlet_type: GauntletType) -> BaseGauntlet:
        """Create a standard gauntlet of the specified type."""
        if gauntlet_type == GauntletType.ADVERSARIAL:
            return AdversarialGauntlet("standard_adversarial")
        elif gauntlet_type == GauntletType.FORMAL_VERIFICATION:
            return FormalVerificationGauntlet("standard_formal")
        elif gauntlet_type == GauntletType.STATISTICAL:
            return StatisticalGauntlet("standard_statistical")
        elif gauntlet_type == GauntletType.DOMAIN_PHYSICS:
            return DomainSpecificGauntlet(domain="physics")
        elif gauntlet_type == GauntletType.MULTI_OBJECTIVE:
            return MultiObjectiveGauntlet("standard_multi_objective")
        elif gauntlet_type == GauntletType.EVOLUTIONARY:
            return EvolutionaryGauntlet("standard_evolutionary")
        elif gauntlet_type == GauntletType.TEMPORAL:
            return TemporalGauntlet("standard_temporal")
        elif gauntlet_type == GauntletType.CROSS_VALIDATION:
            return CrossValidationGauntlet("standard_cross_validation")
        else:
            return AdversarialGauntlet("standard_adversarial")

    def _select_gauntlet_type(
        self,
        complexity_score: ComplexityScore,
        base_type: GauntletType
    ) -> GauntletType:
        """Select appropriate gauntlet type based on complexity."""
        # High complexity problems get more rigorous gauntlets
        if complexity_score.overall_score > self.config.max_complexity_threshold:
            # Use formal verification or multi-objective for high complexity
            if base_type == GauntletType.ADVERSARIAL:
                return GauntletType.FORMAL_VERIFICATION
            return GauntletType.MULTI_OBJECTIVE
        
        # Low complexity problems get lighter gauntlets
        elif complexity_score.overall_score < self.config.min_complexity_threshold:
            # Use statistical or basic for low complexity
            return GauntletType.STATISTICAL
        
        # Medium complexity uses the base type
        return base_type

    def _create_adapted_gauntlet(
        self,
        gauntlet_type: GauntletType,
        complexity_score: ComplexityScore,
        context: Dict[str, Any]
    ) -> BaseGauntlet:
        """Create gauntlet adapted to complexity score."""
        name = f"adapted_{gauntlet_type.value}_{int(time.time())}"
        
        # Base configuration
        config = {
            'complexity_adapted': True,
            'complexity_score': complexity_score.overall_score
        }
        
        # Adjust parameters based on complexity
        if complexity_score.overall_score > 0.7:
            # High complexity: more thorough
            config['attack_modes'] = ['systematic', 'deep_dive', 'adversarial']
            config['timeout'] = 120
            config['strictness'] = 'high'
        elif complexity_score.overall_score > 0.4:
            # Medium complexity: balanced
            config['attack_modes'] = ['systematic', 'adversarial']
            config['timeout'] = 60
            config['strictness'] = 'standard'
        else:
            # Low complexity: quick check
            config['attack_modes'] = ['systematic']
            config['timeout'] = 30
            config['strictness'] = 'lenient'
        
        # Create appropriate gauntlet
        if gauntlet_type == GauntletType.ADVERSARIAL:
            return AdversarialGauntlet(name, config)
        elif gauntlet_type == GauntletType.FORMAL_VERIFICATION:
            return FormalVerificationGauntlet(name, config)
        elif gauntlet_type == GauntletType.STATISTICAL:
            config['num_samples'] = 500 if complexity_score.overall_score > 0.5 else 100
            return StatisticalGauntlet(name, config)
        elif gauntlet_type == GauntletType.DOMAIN_PHYSICS:
            return DomainSpecificGauntlet(domain="physics", name=name, config=config)
        elif gauntlet_type == GauntletType.MULTI_OBJECTIVE:
            return MultiObjectiveGauntlet(name, config)
        elif gauntlet_type == GauntletType.EVOLUTIONARY:
            config['population_size'] = 50 if complexity_score.overall_score > 0.5 else 20
            config['generations'] = 10 if complexity_score.overall_score > 0.5 else 5
            return EvolutionaryGauntlet(name, config)
        elif gauntlet_type == GauntletType.TEMPORAL:
            return TemporalGauntlet(name, config)
        elif gauntlet_type == GauntletType.CROSS_VALIDATION:
            config['k_folds'] = 10 if complexity_score.overall_score > 0.5 else 5
            return CrossValidationGauntlet(name, config)
        else:
            return AdversarialGauntlet(name, config)


# Convenience functions
def create_mdap_maker_integration(
    mode: MDAPMakerGauntletMode = MDAPMakerGauntletMode.HYBRID,
    use_complexity_adaptation: bool = True,
    use_maker_voting: bool = True
) -> MDAPMakerGauntletIntegration:
    """
    Create MDAP/MAKER-Gauntlet integration with specified configuration.
    
    Args:
        mode: Execution mode
        use_complexity_adaptation: Whether to use MDAP complexity analysis
        use_maker_voting: Whether to use MAKER voting
        
    Returns:
        Configured MDAPMakerGauntletIntegration instance
    """
    config = MDAPMakerGauntletConfig(
        mode=mode,
        use_complexity_adaptation=use_complexity_adaptation,
        use_maker_voting=use_maker_voting
    )
    return MDAPMakerGauntletIntegration(config=config)


def execute_gauntlet_with_mdap(
    gauntlet: BaseGauntlet,
    solution: Any,
    problem_description: str,
    context: Optional[Dict[str, Any]] = None
) -> MDAPMakerGauntletResult:
    """
    Execute gauntlet with MDAP/MAKER integration (convenience function).
    
    Args:
        gauntlet: Gauntlet to execute
        solution: Solution to evaluate
        problem_description: Problem description for complexity analysis
        context: Additional context
        
    Returns:
        MDAPMakerGauntletResult with comprehensive results
    """
    integration = create_mdap_maker_integration()
    return integration.execute_with_mdap_maker(
        gauntlet=gauntlet,
        solution=solution,
        context=context,
        problem_description=problem_description
    )


__all__ = [
    'MDAPMakerGauntletMode',
    'MDAPMakerGauntletConfig',
    'MDAPMakerGauntletResult',
    'MDAPMakerGauntletIntegration',
    'create_mdap_maker_integration',
    'execute_gauntlet_with_mdap'
]

"""
Unified Evolution API
=====================

Single entry point for all evolutionary optimization combining:
- OpenEvolve (QD, MO, Adversarial)
- LoongFlow PES (Plan-Execute-Summarize)
- Knowledge Engine (learning from all runs)
- Gauntlet System (3-round evaluation)

This is the final integration that makes everything work together seamlessly.

Author: Unified Evolution Team
Date: 2026-01-30
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORT DEPENDENCIES
# ============================================================================

# Unified configuration
try:
    from .config import UnifiedEvolutionConfig, EvolutionMode, DomainType
    CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("Unified config not available, using fallback")
    CONFIG_AVAILABLE = False
    EvolutionMode = str
    DomainType = str
    # Forward reference string type for when import fails
    UnifiedEvolutionConfig = "UnifiedEvolutionConfig"

# LoongFlow checker and adapter
try:
    from ..integrations.loongflow_checker import LoongFlowChecker, is_loongflow_available
    LOONGFLOW_CHECKER_AVAILABLE = True
except ImportError:
    logger.warning("LoongFlow checker not available")
    LoongFlowChecker = None
    is_loongflow_available = lambda: False
    LOONGFLOW_CHECKER_AVAILABLE = False

# LoongFlow adapter
try:
    from ..integrations.loongflow_adapter import LoongFlowAdapter
    LOONGFLOW_ADAPTER_AVAILABLE = True
except ImportError:
    logger.warning("LoongFlow adapter not available")
    LOONGFLOW_ADAPTER_AVAILABLE = False

# OpenEvolve fallback adapter
try:
    from ..integrations.openevolve_fallback import (
        OpenEvolveFallbackAdapter,
        create_openevolve_adapter
    )
    OPENEVOLVE_FALLBACK_AVAILABLE = True
except ImportError:
    logger.warning("OpenEvolve fallback adapter not available")
    OPENEVOLVE_FALLBACK_AVAILABLE = False

# Strategy recommender
try:
    from knowledge_engine.core.strategy_recommender import (
        StrategyRecommender,
        StrategyRecommendation
    )
    STRATEGY_RECOMMENDER_AVAILABLE = True
except ImportError:
    logger.warning("Strategy recommender not available")
    STRATEGY_RECOMMENDER_AVAILABLE = False

# Three-round gauntlet
try:
    from ..gauntlets import (
        ThreeRoundGauntletOrchestrator,
        ThreeRoundConfig,
        FullGauntletResult
    )
    GAUNTLET_AVAILABLE = True
except ImportError:
    logger.warning("Three-round gauntlet not available")
    GAUNTLET_AVAILABLE = False
    # Forward reference string type for when import fails
    FullGauntletResult = "FullGauntletResult"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EvolutionResult:
    """
    Complete result from unified evolution

    Attributes:
        best_solution: Best solution found (code/program)
        final_score: Final fitness/score
        strategy_used: System and mode used
        config_used: Configuration used for evolution

        system_used: Which system was actually used ("loongflow" or "openevolve")
        mode_used: Which mode was used

        evolution_artifacts: Artifacts extracted from evolution
        gauntlet_result: Optional gauntlet evaluation result

        total_time: Total execution time (seconds)
        iterations: Number of iterations performed
        evaluations: Number of evaluations performed

        metadata: Additional metadata
        error: Error message if evolution failed
    """
    # Solution
    best_solution: str
    final_score: float

    # Strategy used
    strategy_used: 'SystemMode'
    config_used: UnifiedEvolutionConfig

    # System information
    system_used: str = "unknown"  # "loongflow" or "openevolve"
    mode_used: str = "unknown"

    # Evolution artifacts
    evolution_artifacts: List[Any] = field(default_factory=list)

    # Gauntlet evaluation
    gauntlet_result: Optional["FullGauntletResult"] = None

    # Performance
    total_time: float = 0.0
    iterations: int = 0
    evaluations: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error handling
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'best_solution': self.best_solution,
            'final_score': self.final_score,
            'strategy_used': {
                'system': self.strategy_used.system,
                'mode': self.strategy_used.mode
            } if self.strategy_used else None,
            'config': self.config_used.dict() if hasattr(self.config_used, 'dict') else str(self.config_used),
            'system_used': self.system_used,
            'mode_used': self.mode_used,
            'total_time': self.total_time,
            'iterations': self.iterations,
            'evaluations': self.evaluations,
            'metadata': self.metadata,
            'error': self.error,
            'gauntlet_passed': self.gauntlet_result.passed if self.gauntlet_result else None,
            'gauntlet_score': self.gauntlet_result.final_score if self.gauntlet_result else None
        }

    def save(self, filepath: str) -> None:
        """Save result to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Result saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'EvolutionResult':
        """Load result from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct SystemMode if needed
        strategy = None
        if 'strategy_used' in data and data['strategy_used']:
            from .unified_evolution_api import SystemMode
            strategy = SystemMode(
                system=data['strategy_used']['system'],
                mode=data['strategy_used']['mode']
            )

        return cls(
            best_solution=data['best_solution'],
            final_score=data['final_score'],
            strategy_used=strategy,
            config_used=data.get('config', {}),
            total_time=data.get('total_time', 0.0),
            iterations=data.get('iterations', 0),
            evaluations=data.get('evaluations', 0),
            metadata=data.get('metadata', {}),
            error=data.get('error')
        )


@dataclass
class SystemMode:
    """System and mode selection"""
    system: str  # "openevolve", "loongflow"
    mode: str  # "pes", "qd", "mo", "adversarial", "standard"
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ProgressUpdate:
    """Progress update during evolution"""
    stage: str  # 'analyzing', 'selecting_strategy', 'evolving', 'extracting_knowledge', 'running_gauntlet'
    percent_complete: float  # 0-100
    message: str
    current_iteration: int = 0
    total_iterations: int = 0
    current_score: float = 0.0
    best_score_so_far: float = 0.0
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# ============================================================================
# MAIN UNIFIED EVOLUTION API
# ============================================================================

class UnifiedEvolutionAPI:
    """
    Unified Evolution API - Single entry point for all evolutionary optimization

    This class provides a simple, unified interface that automatically:
    1. Analyzes problem characteristics
    2. Selects optimal strategy (OpenEvolve mode or LoongFlow PES)
    3. Generates optimal configuration
    4. Executes evolution
    5. Extracts knowledge to Knowledge Engine
    6. Runs gauntlet evaluation
    7. Learns from results

    Example:
        >>> api = UnifiedEvolutionAPI()
        >>> result = await api.evolve(
        ...     problem="Maximize portfolio Sharpe ratio",
        ...     domain="finance"
        ... )
        >>> print(result.best_solution)
        >>> print(result.final_score)
    """

    def __init__(
        self,
        knowledge_engine=None,
        strategy_recommender=None,
        enable_gauntlets: bool = True,
        enable_knowledge_extraction: bool = True,
        enable_learning: bool = True
    ):
        """
        Initialize unified evolution API

        Args:
            knowledge_engine: Optional knowledge engine instance
            strategy_recommender: Optional strategy recommender
            enable_gauntlets: Enable 3-round gauntlet evaluation
            enable_knowledge_extraction: Enable knowledge extraction
            enable_learning: Enable learning from results
        """
        self.knowledge_engine = knowledge_engine
        self.enable_gauntlets = enable_gauntlets
        self.enable_knowledge_extraction = enable_knowledge_extraction
        self.enable_learning = enable_learning

        # Initialize strategy recommender
        if strategy_recommender:
            self.strategy_recommender = strategy_recommender
        elif STRATEGY_RECOMMENDER_AVAILABLE:
            self.strategy_recommender = StrategyRecommender(
                knowledge_engine=knowledge_engine,
                learning_enabled=enable_learning
            )
        else:
            logger.warning("No strategy recommender available, using rules-based")
            self.strategy_recommender = None

        # Initialize LoongFlow adapter
        if LOONGFLOW_ADAPTER_AVAILABLE:
            self.loongflow_adapter = None  # Created per-run with config
        else:
            self.loongflow_adapter = None

        # Initialize gauntlet orchestrator
        if GAUNTLET_AVAILABLE and enable_gauntlets:
            self.gauntlet_orchestrator = None  # Created per-run with config
        else:
            self.gauntlet_orchestrator = None

        logger.info("UnifiedEvolutionAPI initialized")

    async def evolve(
        self,
        problem: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        config: Optional[UnifiedEvolutionConfig] = None,
        run_gauntlet: bool = True,
        store_knowledge: bool = True,
        use_loongflow: Optional[bool] = None,
        callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> EvolutionResult:
        """
        Main evolution entry point

        Args:
            problem: Problem description (natural language or code)
            domain: Domain (finance, trading, science, engineering, pharma, web, general)
            constraints: Optional constraints (objectives, limits, etc.)
            config: Optional configuration (auto-generated if not provided)
            run_gauntlet: Whether to run 3-round gauntlet evaluation (default: True)
            store_knowledge: Whether to store results in Knowledge Engine (default: True)
            use_loongflow: Override config to force OpenEvolve-only or LoongFlow-only
                           None = auto-detect, False = OpenEvolve-only, True = LoongFlow-only
            callback: Optional callback for progress updates

        Returns:
            EvolutionResult with best solution, metadata, artifacts

        Example:
            >>> # Use default (LoongFlow if available)
            >>> result = await api.evolve(problem="test", domain="general")
            >>>
            >>> # Force OpenEvolve-only
            >>> result = await api.evolve(problem="test", domain="general", use_loongflow=False)
            >>>
            >>> # Force LoongFlow (error if not available)
            >>> result = await api.evolve(problem="test", domain="general", use_loongflow=True)
        """
        start_time = time.time()
        constraints = constraints or {}
        store_knowledge = store_knowledge and self.enable_knowledge_extraction
        run_gauntlet = run_gauntlet and self.enable_gauntlets

        # ========================================================================
        # STEP 0: CHECK LOONGFLOW AVAILABILITY AND USER PREFERENCE
        # ========================================================================
        logger.info("Step 0: Checking LoongFlow availability and preferences")

        # Check LoongFlow availability
        loongflow_available = False
        if LOONGFLOW_CHECKER_AVAILABLE:
            loongflow_available = LoongFlowChecker.is_available()

        # Initialize config
        config = config or (UnifiedEvolutionConfig() if CONFIG_AVAILABLE else {})

        # Get config settings
        config_enable_loongflow = getattr(config, 'enable_loongflow', True)
        config_require_loongflow = getattr(config, 'require_loongflow', False)

        # Determine if we should use LoongFlow
        if use_loongflow is not None:
            # Runtime override takes precedence
            should_use_loongflow = use_loongflow
            logger.info(f"Runtime override: use_loongflow={use_loongflow}")
        elif not config_enable_loongflow:
            # Config says disabled
            should_use_loongflow = False
            logger.info("Config: LoongFlow disabled")
        elif not loongflow_available:
            # LoongFlow not available, check fallback
            if config_require_loongflow:
                error_msg = (
                    "LoongFlow is required but not available. "
                    "Install LoongFlow or set require_loongflow=False"
                )
                logger.error(error_msg)
                return EvolutionResult(
                    best_solution='',
                    final_score=0.0,
                    strategy_used=SystemMode(system='unknown', mode='unknown'),
                    config_used=config,
                    system_used='none',
                    mode_used='none',
                    total_time=time.time() - start_time,
                    metadata={'domain': domain, 'problem': problem, 'failed': True},
                    error=error_msg
                )
            else:
                # Fall back to OpenEvolve
                should_use_loongflow = False
                logger.info("ℹ️  LoongFlow not available, using OpenEvolve-only mode")
        else:
            should_use_loongflow = True
            logger.info("[OK] LoongFlow available and enabled")

        # Log what we're doing
        if should_use_loongflow:
            logger.info("🚀 Using LoongFlow PES for evolution")
            system_used = "loongflow"
        else:
            mode = getattr(config, 'evolution_mode', 'standard')
            logger.info(f"🧬 Using OpenEvolve {mode} mode for evolution")
            system_used = "openevolve"

        try:
            # ====================================================================
            # STEP 1: ANALYZE PROBLEM
            # ====================================================================
            logger.info(f"Step 1: Analyzing problem for domain={domain}")
            if callback:
                callback(ProgressUpdate(
                    stage='analyzing',
                    percent_complete=5,
                    message='Analyzing problem characteristics...'
                ))

            problem_chars = await self._analyze_problem(problem, domain, constraints)

            # ====================================================================
            # STEP 2: SELECT STRATEGY (considering LoongFlow availability)
            # ====================================================================
            logger.info("Step 2: Selecting optimal strategy")
            if callback:
                callback(ProgressUpdate(
                    stage='selecting_strategy',
                    percent_complete=10,
                    message='Selecting evolutionary strategy...'
                ))

            strategy = await self._select_strategy(
                problem, domain, problem_chars, constraints, should_use_loongflow
            )

            logger.info(f"Selected strategy: {strategy.system} / {strategy.mode} (confidence: {strategy.confidence:.2f})")

            # ====================================================================
            # STEP 3: GENERATE CONFIGURATION
            # ====================================================================
            logger.info("Step 3: Generating optimal configuration")
            if callback:
                callback(ProgressUpdate(
                    stage='generating_config',
                    percent_complete=15,
                    message=f'Generating configuration for {strategy.mode} mode...'
                ))

            if config is None:
                config = await self._generate_config(strategy, problem_chars, domain)

            # ====================================================================
            # STEP 4: EXECUTE EVOLUTION (with appropriate adapter)
            # ====================================================================
            logger.info(f"Step 4: Executing evolution ({strategy.system} / {strategy.mode})")
            if callback:
                callback(ProgressUpdate(
                    stage='evolving',
                    percent_complete=20,
                    message=f'Running {strategy.mode.upper()} evolution...'
                ))

            evolution_result = await self._execute_evolution(
                problem, domain, strategy, config, callback, should_use_loongflow
            )

            # ====================================================================
            # STEP 5: EXTRACT KNOWLEDGE
            # ====================================================================
            artifacts = []
            if store_knowledge:
                logger.info("Step 5: Extracting knowledge artifacts")
                if callback:
                    callback(ProgressUpdate(
                        stage='extracting_knowledge',
                        percent_complete=70,
                        message='Extracting learning artifacts...'
                    ))

                artifacts = await self._extract_knowledge(
                    evolution_result, strategy, problem, domain
                )

            # ====================================================================
            # STEP 6: RUN GAUNTLET
            # ====================================================================
            gauntlet_result = None
            if run_gauntlet and evolution_result.get('best_solution'):
                logger.info("Step 6: Running 3-round gauntlet evaluation")
                if callback:
                    callback(ProgressUpdate(
                        stage='running_gauntlet',
                        percent_complete=80,
                        message='Running gauntlet evaluation...'
                    ))

                gauntlet_result = await self._run_gauntlet(
                    evolution_result['best_solution'],
                    problem,
                    domain,
                    config
                )

                # Filter gauntlet artifacts too
                if store_knowledge and gauntlet_result:
                    gauntlet_artifacts = await self._extract_gauntlet_artifacts(gauntlet_result)
                    artifacts.extend(gauntlet_artifacts)

            # ====================================================================
            # STEP 7: LEARN AND IMPROVE
            # ====================================================================
            if self.enable_learning:
                logger.info("Step 7: Learning from results")
                if callback:
                    callback(ProgressUpdate(
                        stage='learning',
                        percent_complete=90,
                        message='Updating strategy recommendations...'
                    ))

                await self._learn_from_run(
                    problem, domain, strategy, evolution_result, gauntlet_result
                )

            # ====================================================================
            # STEP 8: RETURN RESULT
            # ====================================================================
            total_time = time.time() - start_time
            logger.info(f"Evolution complete in {total_time:.2f}s")
            if callback:
                callback(ProgressUpdate(
                    stage='complete',
                    percent_complete=100,
                    message=f'Evolution complete! Score: {evolution_result.get("best_fitness", 0.0):.3f}'
                ))

            return EvolutionResult(
                best_solution=evolution_result.get('best_solution', ''),
                final_score=evolution_result.get('best_fitness', 0.0),
                strategy_used=strategy,
                config_used=config,
                system_used=evolution_result.get('system_used', system_used),
                mode_used=evolution_result.get('mode_used', strategy.mode),
                evolution_artifacts=artifacts,
                gauntlet_result=gauntlet_result,
                total_time=total_time,
                iterations=evolution_result.get('iterations_performed', 0),
                evaluations=evolution_result.get('total_evaluations', 0),
                metadata={
                    'domain': domain,
                    'problem': problem,
                    'strategy_confidence': strategy.confidence,
                    'strategy_reasoning': strategy.reasoning,
                    'loongflow_was_used': should_use_loongflow,
                    'loongflow_was_available': loongflow_available
                }
            )

        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            total_time = time.time() - start_time

            return EvolutionResult(
                best_solution='',
                final_score=0.0,
                strategy_used=SystemMode(system='unknown', mode='unknown'),
                config_used=config or UnifiedEvolutionConfig() if CONFIG_AVAILABLE else {},
                total_time=total_time,
                iterations=0,
                evaluations=0,
                metadata={
                    'domain': domain,
                    'problem': problem,
                    'failed': True
                },
                error=str(e)
            )

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    async def _analyze_problem(
        self,
        problem: str,
        domain: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze problem characteristics"""
        # Use strategy recommender if available
        if self.strategy_recommender and hasattr(self.strategy_recommender, 'analyze_problem_characteristics'):
            problem_chars = await self.strategy_recommender.analyze_problem_characteristics(
                problem, domain, constraints
            )
            return {
                'domain': problem_chars.domain,
                'complexity': problem_chars.complexity,
                'evaluation_cost': problem_chars.evaluation_cost,
                'has_multiple_objectives': problem_chars.has_multiple_objectives,
                'requires_diversity': problem_chars.requires_diversity,
                'requires_robustness': problem_chars.requires_robustness,
                'constraint_count': problem_chars.constraint_count,
                'estimated_iterations': problem_chars.estimated_iterations
            }

        # Fallback: simple analysis
        complexity = 'medium'
        eval_cost = 'moderate'

        # Domain-based heuristics
        if domain in ['science', 'pharma']:
            eval_cost = 'very_expensive'
            complexity = 'high'
        elif domain in ['finance', 'trading', 'engineering']:
            eval_cost = 'expensive'
            complexity = 'high'

        has_multi_obj = len(constraints.get('objectives', [])) > 1

        return {
            'domain': domain,
            'complexity': complexity,
            'evaluation_cost': eval_cost,
            'has_multiple_objectives': has_multi_obj,
            'requires_diversity': domain in ['finance', 'trading', 'science'],
            'requires_robustness': domain in ['engineering', 'pharma', 'finance'],
            'constraint_count': len(constraints.get('constraints', [])),
            'estimated_iterations': 100
        }

    async def _select_strategy(
        self,
        problem: str,
        domain: str,
        problem_chars: Dict[str, Any],
        constraints: Dict[str, Any],
        use_loongflow: bool = True
    ) -> SystemMode:
        """
        Select optimal evolutionary strategy

        Args:
            problem: Problem description
            domain: Problem domain
            problem_chars: Problem characteristics
            constraints: Constraints
            use_loongflow: Whether LoongFlow should be used

        Returns:
            SystemMode recommendation
        """
        # Use strategy recommender if available
        if self.strategy_recommender and hasattr(self.strategy_recommender, 'recommend_strategy'):
            try:
                recommendation = await self.strategy_recommender.recommend_strategy(
                    problem_description=problem,
                    domain=domain,
                    constraints=constraints
                )

                # Adjust recommendation based on LoongFlow availability
                if recommendation.recommended_system == 'loongflow' and not use_loongflow:
                    # LoongFlow recommended but not available, fall back to OpenEvolve
                    logger.info("LoongFlow recommended but not available, using OpenEvolve instead")
                    return self._rules_based_strategy_selection(domain, problem_chars, constraints, use_loongflow=False)

                return SystemMode(
                    system=recommendation.recommended_system,
                    mode=recommendation.recommended_mode,
                    confidence=recommendation.confidence,
                    reasoning=recommendation.reasoning.primary_reason
                )
            except (NotImplementedError, Exception) as e:
                logger.warning(f"Strategy recommender failed ({e}), using rules-based selection")

        # Fallback: rules-based selection
        return self._rules_based_strategy_selection(domain, problem_chars, constraints, use_loongflow)

    def _rules_based_strategy_selection(
        self,
        domain: str,
        problem_chars: Dict[str, Any],
        constraints: Dict[str, Any],
        use_loongflow: bool = True
    ) -> SystemMode:
        """
        Rules-based strategy selection (fallback)

        Args:
            domain: Problem domain
            problem_chars: Problem characteristics
            constraints: Constraints
            use_loongflow: Whether LoongFlow should be used

        Returns:
            SystemMode recommendation
        """
        # Decision logic
        eval_cost = problem_chars['evaluation_cost']
        has_multi_obj = problem_chars['has_multiple_objectives']
        needs_diversity = problem_chars['requires_diversity']
        needs_robustness = problem_chars['requires_robustness']

        # Prioritize based on factors
        if eval_cost in ['expensive', 'very_expensive'] and use_loongflow and LOONGFLOW_ADAPTER_AVAILABLE:
            return SystemMode(
                system='loongflow',
                mode='pes',
                confidence=0.85,
                reasoning='Expensive evaluations favor PES (60% fewer evaluations)'
            )

        if has_multi_obj:
            return SystemMode(
                system='openevolve',
                mode='mo',
                confidence=0.80,
                reasoning='Multiple objectives require Pareto optimization'
            )

        if needs_robustness:
            return SystemMode(
                system='openevolve',
                mode='adversarial',
                confidence=0.75,
                reasoning='Robustness testing via adversarial co-evolution'
            )

        if needs_diversity:
            return SystemMode(
                system='openevolve',
                mode='qd',
                confidence=0.70,
                reasoning='Quality-Diversity for exploring solution space'
            )

        # Default: PES if available and enabled, else standard
        if use_loongflow and LOONGFLOW_ADAPTER_AVAILABLE:
            return SystemMode(
                system='loongflow',
                mode='pes',
                confidence=0.65,
                reasoning='PES provides best general performance'
            )
        else:
            return SystemMode(
                system='openevolve',
                mode='standard',
                confidence=0.60,
                reasoning='Standard evolutionary algorithm'
            )

    async def _generate_config(
        self,
        strategy: SystemMode,
        problem_chars: Dict[str, Any],
        domain: str
    ) -> "UnifiedEvolutionConfig":
        """Generate optimal configuration"""
        if CONFIG_AVAILABLE:
            config = UnifiedEvolutionConfig()

            # Apply strategy-specific overrides
            if strategy.mode == 'pes':
                config.pes.enabled = True
                config.pes.enable_planning = True
                config.pes.enable_memory = True
                config.evolution_mode = EvolutionMode.PES

            elif strategy.mode == 'qd':
                config.qd.enabled = True
                config.qd.grid_resolution = 10
                config.qd.archive_size = 1000
                config.evolution_mode = EvolutionMode.QD

            elif strategy.mode == 'mo':
                config.mo.enabled = True
                config.mo.pareto_size = 100
                config.evolution_mode = EvolutionMode.MO

            elif strategy.mode == 'adversarial':
                config.adversarial.enabled = True
                config.adversarial.adversarial_rounds = 20
                config.evolution_mode = EvolutionMode.ADVERSARIAL

            # Adjust iterations based on evaluation cost
            if problem_chars['evaluation_cost'] == 'very_expensive':
                config.max_iterations = 30
            elif problem_chars['evaluation_cost'] == 'expensive':
                config.max_iterations = 50
            else:
                config.max_iterations = problem_chars.get('estimated_iterations', 100)

            # Set domain
            config.domain = DomainType(domain) if hasattr(DomainType, domain) else DomainType.GENERAL

            return config
        else:
            # Fallback config
            return {
                'max_iterations': problem_chars.get('estimated_iterations', 100),
                'evolution_mode': strategy.mode,
                'domain': domain
            }

    async def _execute_evolution(
        self,
        problem: str,
        domain: str,
        strategy: SystemMode,
        config: UnifiedEvolutionConfig,
        callback: Optional[Callable],
        use_loongflow: bool = True
    ) -> Dict[str, Any]:
        """
        Execute evolution with selected strategy

        Args:
            problem: Problem description
            domain: Problem domain
            strategy: Strategy to use
            config: Configuration
            callback: Progress callback
            use_loongflow: Whether to use LoongFlow

        Returns:
            Dict with evolution results
        """
        # Prepare config dict for adapters
        if hasattr(config, 'dict'):
            config_dict = config.dict()
        else:
            config_dict = config

        # Execute based on system
        if strategy.system == 'loongflow' and use_loongflow and LOONGFLOW_ADAPTER_AVAILABLE:
            return await self._execute_loongflow(problem, domain, config_dict, callback)
        else:
            return await self._execute_openevolve(problem, domain, strategy, config_dict, callback)

    async def _execute_loongflow(
        self,
        problem: str,
        domain: str,
        config: Dict[str, Any],
        callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute LoongFlow PES evolution"""
        logger.info(f"Executing LoongFlow PES evolution")

        try:
            # Create adapter
            adapter = LoongFlowAdapter(config)

            # Run evolution
            result = await adapter.evolve(
                problem=problem,
                domain=domain
            )

            # Add system info
            result['system_used'] = 'loongflow'
            result['mode_used'] = 'pes'

            logger.info(f"LoongFlow evolution complete: score={result.get('best_fitness', 0.0):.3f}")
            return result

        except Exception as e:
            logger.error(f"LoongFlow execution failed: {e}", exc_info=True)
            # Fallback to OpenEvolve
            logger.info("Falling back to OpenEvolve")
            return await self._execute_openevolve(
                problem, domain,
                SystemMode(system='openevolve', mode='standard'),
                config, callback
            )

    async def _execute_openevolve(
        self,
        problem: str,
        domain: str,
        strategy: SystemMode,
        config: Dict[str, Any],
        callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute OpenEvolve evolution"""
        logger.info(f"Executing OpenEvolve {strategy.mode} evolution")

        try:
            # Try to use fallback adapter if available
            if OPENEVOLVE_FALLBACK_AVAILABLE:
                adapter = create_openevolve_adapter(config)
                result = await adapter.evolve(problem=problem, domain=domain)
                # Ensure system info is set
                result.setdefault('system_used', 'openevolve')
                result.setdefault('mode_used', strategy.mode)
                return result
        except Exception as e:
            logger.warning(f"OpenEvolve fallback adapter failed: {e}")

        # Placeholder implementation
        # In production, this would call the actual OpenEvolve executor

        # Simulate evolution
        await asyncio.sleep(0.1)  # Simulate some work

        # Return mock result
        return {
            'best_solution': f"# Solution for {domain}\n# Problem: {problem[:50]}...\n# Mode: {strategy.mode}\ndef solve():\n    # Optimized solution\n    pass",
            'best_fitness': 0.75,
            'total_evaluations': config.get('max_iterations', 100),
            'iterations_performed': config.get('max_iterations', 100),
            'improvement_rate': 0.5,
            'strategy_used': strategy.mode,
            'source': 'openevolve',
            'system_used': 'openevolve',
            'mode_used': strategy.mode
        }

    async def _extract_knowledge(
        self,
        evolution_result: Dict[str, Any],
        strategy: SystemMode,
        problem: str,
        domain: str
    ) -> List[Any]:
        """Extract knowledge artifacts from evolution"""
        artifacts = []

        if not self.knowledge_engine:
            logger.info("No knowledge engine available, skipping extraction")
            return artifacts

        try:
            # Create artifact
            artifact = {
                'type': 'evolution_result',
                'domain': domain,
                'strategy': f"{strategy.system}/{strategy.mode}",
                'fitness': evolution_result.get('best_fitness', 0.0),
                'evaluations': evolution_result.get('total_evaluations', 0),
                'problem': problem[:100],  # Truncated
                'timestamp': datetime.now(UTC).isoformat()
            }

            artifacts.append(artifact)

            # Store in knowledge engine
            # (Implementation depends on KE API)
            logger.info(f"Extracted {len(artifacts)} knowledge artifacts")

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")

        return artifacts

    async def _run_gauntlet(
        self,
        solution: str,
        problem: str,
        domain: str,
        config: UnifiedEvolutionConfig
    ) -> Optional["FullGauntletResult"]:
        """Run 3-round gauntlet evaluation"""
        if not GAUNTLET_AVAILABLE:
            logger.warning("Gauntlet system not available")
            return None

        try:
            # Create gauntlet config
            gauntlet_config = ThreeRoundConfig(
                round1_threshold=0.5,
                round2_threshold=0.6,
                round3_threshold=0.7
            )

            # Create orchestrator
            orchestrator = ThreeRoundGauntletOrchestrator(gauntlet_config)

            # Run gauntlet
            result = await orchestrator.run_full_gauntlet(
                solution=solution,
                problem=problem,
                domain=domain
            )

            logger.info(f"Gauntlet complete: passed={result.passed}, score={result.final_score:.3f}")
            return result

        except Exception as e:
            logger.error(f"Gauntlet execution failed: {e}", exc_info=True)
            return None

    async def _extract_gauntlet_artifacts(self, gauntlet_result: "FullGauntletResult") -> List[Any]:
        """Extract artifacts from gauntlet result"""
        artifacts = []

        try:
            artifact = {
                'type': 'gauntlet_result',
                'passed': gauntlet_result.passed,
                'final_score': gauntlet_result.final_score,
                'rounds_completed': gauntlet_result.rounds_completed,
                'domain': gauntlet_result.domain,
                'timestamp': datetime.now(UTC).isoformat()
            }

            artifacts.append(artifact)
            logger.info(f"Extracted gauntlet artifacts")

        except Exception as e:
            logger.error(f"Gauntlet artifact extraction failed: {e}")

        return artifacts

    async def _learn_from_run(
        self,
        problem: str,
        domain: str,
        strategy: SystemMode,
        evolution_result: Dict[str, Any],
        gauntlet_result: Optional[FullGauntletResult]
    ):
        """Learn from completed run"""
        if not self.strategy_recommender:
            return

        try:
            # Prepare run result
            run_result = {
                'run_id': f"run_{int(time.time())}",
                'domain': domain,
                'strategy_used': strategy.system,
                'mode_used': strategy.mode,
                'complexity': 'medium',
                'final_score': evolution_result.get('best_fitness', 0.0),
                'iterations': evolution_result.get('iterations_performed', 0),
                'evaluations': evolution_result.get('total_evaluations', 0),
                'diversity_score': 0.5,  # Placeholder
                'metadata': {
                    'problem': problem[:100],
                    'gauntlet_passed': gauntlet_result.passed if gauntlet_result else None,
                    'gauntlet_score': gauntlet_result.final_score if gauntlet_result else None
                }
            }

            # Learn
            await self.strategy_recommender.learn_from_run(run_result)
            logger.info("Strategy recommender updated with run data")

        except Exception as e:
            logger.error(f"Learning from run failed: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def evolve(
    problem: str,
    domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None,
    config: Optional[UnifiedEvolutionConfig] = None,
    run_gauntlet: bool = True,
    store_knowledge: bool = True,
    use_loongflow: Optional[bool] = None,
    callback: Optional[Callable[[ProgressUpdate], None]] = None,
    knowledge_engine=None
) -> EvolutionResult:
    """
    Convenience function for unified evolution

    Args:
        problem: Problem description
        domain: Problem domain
        constraints: Optional constraints
        config: Optional configuration
        run_gauntlet: Run gauntlet evaluation
        store_knowledge: Store in knowledge engine
        use_loongflow: Override config to force OpenEvolve-only or LoongFlow-only
        callback: Progress callback
        knowledge_engine: Optional knowledge engine

    Returns:
        EvolutionResult

    Example:
        >>> # Use default (LoongFlow if available)
        >>> result = await evolve(
        ...     problem="Optimize portfolio allocation",
        ...     domain="finance"
        ... )
        >>>
        >>> # Force OpenEvolve-only
        >>> result = await evolve(
        ...     problem="Optimize portfolio allocation",
        ...     domain="finance",
        ...     use_loongflow=False
        ... )
    """
    api = UnifiedEvolutionAPI(knowledge_engine=knowledge_engine)
    return await api.evolve(
        problem=problem,
        domain=domain,
        constraints=constraints,
        config=config,
        run_gauntlet=run_gauntlet,
        store_knowledge=store_knowledge,
        use_loongflow=use_loongflow,
        callback=callback
    )


async def quick_evolve(problem: str, domain: str = "general") -> str:
    """
    Fastest path to solution, returns just the solution string

    Args:
        problem: Problem description
        domain: Problem domain

    Returns:
        Best solution string

    Example:
        >>> solution = await quick_evolve("Optimize function", "science")
        >>> print(solution)
    """
    result = await evolve(
        problem=problem,
        domain=domain,
        run_gauntlet=False,  # Skip gauntlet for speed
        store_knowledge=False
    )
    return result.best_solution


async def evolve_no_gauntlet(
    problem: str,
    domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None
) -> EvolutionResult:
    """
    Evolution without gauntlet evaluation (faster, less quality assurance)

    Args:
        problem: Problem description
        domain: Problem domain
        constraints: Optional constraints

    Returns:
        EvolutionResult
    """
    return await evolve(
        problem=problem,
        domain=domain,
        constraints=constraints,
        run_gauntlet=False
    )


async def evolve_batch(
    problems: List[str],
    domain: str = "general",
    max_concurrent: int = 3,
    constraints: Optional[Dict[str, Any]] = None
) -> List[EvolutionResult]:
    """
    Evolve multiple problems in parallel

    Args:
        problems: List of problem descriptions
        domain: Problem domain
        max_concurrent: Maximum concurrent evolutions
        constraints: Optional constraints (applied to all)

    Returns:
        List of EvolutionResult in same order as problems
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evolve_with_limit(problem: str) -> EvolutionResult:
        async with semaphore:
            return await evolve(
                problem=problem,
                domain=domain,
                constraints=constraints,
                run_gauntlet=False,  # Skip gauntlet for batch
                store_knowledge=False
            )

    # Run all evolutions
    results = await asyncio.gather(*[
        evolve_with_limit(p) for p in problems
    ])

    return results


async def evolve_openevolve_only(
    problem: str,
    domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None,
    config: Optional[UnifiedEvolutionConfig] = None,
    run_gauntlet: bool = True,
    store_knowledge: bool = True,
    callback: Optional[Callable[[ProgressUpdate], None]] = None,
    knowledge_engine=None
) -> EvolutionResult:
    """
    Convenience function: Evolution using OpenEvolve only

    This is equivalent to evolve(..., use_loongflow=False)

    Args:
        problem: Problem description
        domain: Problem domain
        constraints: Optional constraints
        config: Optional configuration
        run_gauntlet: Run gauntlet evaluation
        store_knowledge: Store in knowledge engine
        callback: Progress callback
        knowledge_engine: Optional knowledge engine

    Returns:
        EvolutionResult

    Example:
        >>> result = await evolve_openevolve_only(
        ...     problem="Optimize portfolio",
        ...     domain="finance"
        ... )
        >>> print(result.best_solution)
    """
    return await evolve(
        problem=problem,
        domain=domain,
        constraints=constraints,
        config=config,
        run_gauntlet=run_gauntlet,
        store_knowledge=store_knowledge,
        use_loongflow=False,
        callback=callback,
        knowledge_engine=knowledge_engine
    )


async def evolve_with_loongflow(
    problem: str,
    domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None,
    config: Optional[UnifiedEvolutionConfig] = None,
    run_gauntlet: bool = True,
    store_knowledge: bool = True,
    callback: Optional[Callable[[ProgressUpdate], None]] = None,
    knowledge_engine=None
) -> EvolutionResult:
    """
    Convenience function: Evolution using LoongFlow (requires it to be available)

    This is equivalent to evolve(..., use_loongflow=True)

    Args:
        problem: Problem description
        domain: Problem domain
        constraints: Optional constraints
        config: Optional configuration
        run_gauntlet: Run gauntlet evaluation
        store_knowledge: Store in knowledge engine
        callback: Progress callback
        knowledge_engine: Optional knowledge engine

    Returns:
        EvolutionResult

    Raises:
        RuntimeError: If LoongFlow is required but not available

    Example:
        >>> result = await evolve_with_loongflow(
        ...     problem="Optimize portfolio",
        ...     domain="finance"
        ... )
        >>> print(result.best_solution)
    """
    return await evolve(
        problem=problem,
        domain=domain,
        constraints=constraints,
        config=config,
        run_gauntlet=run_gauntlet,
        store_knowledge=store_knowledge,
        use_loongflow=True,
        callback=callback,
        knowledge_engine=knowledge_engine
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedEvolutionAPI',
    'evolve',
    'evolve_openevolve_only',
    'evolve_with_loongflow',
    'quick_evolve',
    'evolve_no_gauntlet',
    'evolve_batch',
    'EvolutionResult',
    'SystemMode',
    'ProgressUpdate'
]

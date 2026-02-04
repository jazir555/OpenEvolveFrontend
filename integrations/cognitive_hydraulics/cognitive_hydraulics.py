"""Cognitive Hydraulics - Main orchestration engine.

Coordinates Soar, ACT-R, Pressure Valve, and Evolutionary fallback.

Architecture:
    - CognitiveHydraulicsEngine: Main engine class
    - ReasoningSession: Session for single problem
    - SystemOrchestrator: Coordinates subsystems

Flow:
    1. Start with Soar (System 2) symbolic reasoning
    2. Monitor pressure via Pressure Valve
    3. If pressure builds, switch to ACT-R (System 1)
    4. If ACT-R fails or pressure >= 0.9, use Evolutionary
    5. Learn successful resolutions via Chunking

Configuration:
    - llm_model: "qwen3:8b" or similar
    - soar_to_actr_depth: 3
    - actr_to_evo_pressure: 0.9
    - enable_chunking: True
    - max_reasoning_time_ms: 30000
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
from enum import Enum

from .config import CognitiveHydraulicsConfig
from .soar_engine import (
    SoarEngine, SoarOperator, SoarRule, SoarState,
    Impasse, ImpasseType
)
from .actr_engine import (
    ACTREngine, ACTRProduction, ACTRChunk, UtilityCalculator
)
from .pressure_valve import (
    PressureValve, SystemType, PressureMetrics
)
from .llm_intuition import LLMIntuitionEngine, SuccessRating
from .evolutionary_fallback import EvolutionarySolver, Individual, SolutionType
from .chunking_system import ChunkingEngine, Chunk

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result of a reasoning session."""
    success: bool
    solution: Any = None
    solution_type: str = ""
    reasoning_trace: List[Dict] = field(default_factory=list)
    systems_used: List[str] = field(default_factory=list)
    
    # Metrics
    total_time_ms: float = 0.0
    cycles_soar: int = 0
    cycles_actr: int = 0
    generations_evo: int = 0
    
    # Learning
    chunks_learned: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "solution": self.solution,
            "solution_type": self.solution_type,
            "systems_used": self.systems_used,
            "total_time_ms": self.total_time_ms,
            "cycles_soar": self.cycles_soar,
            "cycles_actr": self.cycles_actr,
            "generations_evo": self.generations_evo,
            "chunks_learned": self.chunks_learned,
        }


class SystemOrchestrator:
    """Coordinates the cognitive subsystems."""
    
    def __init__(
        self,
        soar: SoarEngine,
        actr: ACTREngine,
        pressure_valve: PressureValve,
        llm_intuition: LLMIntuitionEngine,
        evolutionary: EvolutionarySolver,
        chunking: ChunkingEngine
    ):
        self.soar = soar
        self.actr = actr
        self.pressure_valve = pressure_valve
        self.llm_intuition = llm_intuition
        self.evolutionary = evolutionary
        self.chunking = chunking
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup system switching callbacks."""
        self.pressure_valve.register_callbacks(
            on_switch_to_actr=self._on_switch_to_actr,
            on_switch_to_evolutionary=self._on_switch_to_evolutionary,
            on_switch_to_soar=self._on_switch_to_soar
        )
    
    def _on_switch_to_actr(self):
        """Handle switch to ACT-R."""
        logger.info("Switching to ACT-R")
        
        # Transfer state from Soar to ACT-R
        soar_state = self.soar.get_current_state()
        if soar_state:
            # Convert Soar state to ACT-R context
            context = soar_state.get_all_attributes()
            self.actr.set_goal(soar_state.goal)
    
    def _on_switch_to_evolutionary(self):
        """Handle switch to Evolutionary."""
        logger.info("Switching to Evolutionary")
    
    def _on_switch_to_soar(self):
        """Handle switch to Soar."""
        logger.info("Switching to Soar")


class ReasoningSession:
    """Session for solving a single problem."""
    
    def __init__(
        self,
        session_id: str,
        problem: Dict[str, Any],
        goal: Dict[str, Any],
        orchestrator: SystemOrchestrator,
        config: CognitiveHydraulicsConfig
    ):
        self.session_id = session_id
        self.problem = problem
        self.goal = goal
        self.orchestrator = orchestrator
        self.config = config
        
        # Result tracking
        self.result = ReasoningResult(success=False)
        self.trace: List[Dict] = []
        self.systems_used: set = set()
        
        # Timing
        self.start_time: Optional[float] = None
        self.current_system: SystemType = SystemType.SOAR
    
    def start(self):
        """Start the reasoning session."""
        self.start_time = time.time() * 1000
        self.orchestrator.pressure_valve.start_monitoring()
        
        self._log_event("session_start", {
            "problem": self.problem,
            "goal": self.goal
        })
    
    def run_soar_cycle(self) -> tuple[bool, Optional[Impasse]]:
        """Execute System 2 cycle."""
        self.systems_used.add("soar")
        
        success, impasse = self.orchestrator.soar.run_decision_cycle()
        
        self.result.cycles_soar += 1
        
        if impasse:
            self._log_event("impasse_detected", impasse.to_dict())
            self.orchestrator.pressure_valve.record_impasse(impasse)
        
        return success, impasse
    
    def run_actr_cycle(self) -> Optional[ACTRProduction]:
        """Execute System 1 cycle."""
        self.systems_used.add("actr")
        
        context = self._build_actr_context()
        selected = self.orchestrator.actr.run_cycle(context)
        
        self.result.cycles_actr += 1
        
        if selected:
            self._log_event("actr_operator_selected", {
                "operator": selected.name,
                "utility": selected.utility
            })
            self.orchestrator.actr.update_history(selected.production_id)
        else:
            self._log_event("actr_no_operator", {})
            self.orchestrator.pressure_valve.record_failure()
        
        return selected
    
    def run_evolutionary_fallback(self) -> Optional[Individual]:
        """Execute GA."""
        self.systems_used.add("evolutionary")
        
        # Initialize population based on problem
        self.orchestrator.evolutionary.initialize_population(
            size=self.config.evolutionary.population_size,
            problem=self.problem
        )
        
        # Evolve
        best = self.orchestrator.evolutionary.evolve(
            generations=self.config.evolutionary.max_generations
        )
        
        self.result.generations_evo = self.orchestrator.evolutionary.generation
        
        if best:
            self._log_event("evolutionary_solution", {
                "fitness": best.fitness,
                "syntax_correct": best.syntax_correct,
                "runtime_success": best.runtime_success
            })
        
        return best
    
    def handle_impasse(self, impasse: Impasse) -> Dict[str, Any]:
        """Process impasse and attempt resolution."""
        self._log_event("handling_impasse", impasse.to_dict())
        
        # Try to create subgoal
        subgoal = self.orchestrator.soar.create_subgoal(impasse)
        
        return {
            "subgoal_created": subgoal is not None,
            "subgoal_id": subgoal.state_id if subgoal else None
        }
    
    def learn_from_success(self, impasse: Optional[Impasse], resolution: Dict[str, Any]):
        """Trigger chunking."""
        if not self.config.soar.enable_chunking:
            return
        
        if not impasse:
            return
        
        # Create chunk
        context = self._build_chunk_context()
        chunk = self.orchestrator.chunking.create_chunk(impasse, resolution, context)
        
        if chunk:
            self.result.chunks_learned.append(chunk.chunk_id)
            self._log_event("chunk_created", chunk.to_dict())
    
    def check_system_switch(self) -> SystemType:
        """Check if we need to switch systems."""
        # Update pressure metrics
        context = self._build_pressure_context()
        pressure = self.orchestrator.pressure_valve.compute_pressure({}, context)
        
        # Check for switch
        soar_state = {}
        actr_state = {}
        actr_failure = self.result.cycles_actr > 0 and not self.orchestrator.actr.get_stats()["declarative_chunks"]
        
        new_system = self.orchestrator.pressure_valve.check_and_switch(
            soar_state, actr_state, actr_failure
        )
        
        if new_system != self.current_system:
            self._log_event("system_switch", {
                "from": self.current_system.value,
                "to": new_system.value,
                "pressure": pressure
            })
            self.current_system = new_system
        
        return new_system
    
    def finalize(self, success: bool, solution: Any = None):
        """Finalize the session."""
        elapsed = (time.time() * 1000) - self.start_time if self.start_time else 0
        
        self.result.success = success
        self.result.solution = solution
        self.result.total_time_ms = elapsed
        self.result.systems_used = list(self.systems_used)
        self.result.reasoning_trace = self.trace
        
        self._log_event("session_end", {
            "success": success,
            "total_time_ms": elapsed
        })
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the trace."""
        self.trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": data,
            "system": self.current_system.value
        })
    
    def _build_actr_context(self) -> Dict[str, Any]:
        """Build context for ACT-R."""
        state = self.orchestrator.soar.get_current_state()
        if state:
            return state.get_all_attributes()
        return {}
    
    def _build_pressure_context(self) -> Dict[str, Any]:
        """Build context for pressure monitoring."""
        return {
            "subgoal_depth": self.orchestrator.soar.subgoal_manager.get_current_depth(),
            "cycle_count": self.result.cycles_soar + self.result.cycles_actr,
            "impasse_count": self.orchestrator.soar.impasse_detector.impasse_count,
            "ambiguity_score": 0,  # Would be calculated from available operators
            "memory_load": len(self.orchestrator.soar.working_memory.states)
        }
    
    def _build_chunk_context(self) -> Dict[str, Any]:
        """Build context for chunk creation."""
        return {
            "problem_description": self.problem,
            "goal": self.goal,
            "session_id": self.session_id
        }


class CognitiveHydraulicsEngine:
    """
    Main Cognitive Hydraulics Engine.
    
    Coordinates Soar, ACT-R, Pressure Valve, and Evolutionary fallback.
    """
    
    def __init__(self, config: Optional[CognitiveHydraulicsConfig] = None):
        self.config = config or CognitiveHydraulicsConfig()
        
        # Validate configuration
        validation = self.config.validate()
        if not validation["valid"]:
            raise ValueError(f"Invalid configuration: {validation['errors']}")
        
        # Initialize subsystems
        self.soar = SoarEngine(self.config.soar)
        self.actr = ACTREngine(self.config.actr)
        self.pressure_valve = PressureValve(self.config.pressure_valve)
        self.llm_intuition = LLMIntuitionEngine(self.config.llm)
        self.evolutionary = EvolutionarySolver(self.config.evolutionary)
        self.chunking = ChunkingEngine(self.soar.production_system)
        
        # Setup orchestrator
        self.orchestrator = SystemOrchestrator(
            self.soar,
            self.actr,
            self.pressure_valve,
            self.llm_intuition,
            self.evolutionary,
            self.chunking
        )
        
        # Setup LLM callbacks for ACT-R
        self._setup_llm_callbacks()
        
        # Stats
        self.session_count = 0
        self.success_count = 0
    
    def _setup_llm_callbacks(self):
        """Setup LLM estimation callbacks for ACT-R."""
        def prob_estimator(operator, goal):
            return self.llm_intuition.estimate_probability(operator, goal, {})
        
        def cost_estimator(operator):
            return self.llm_intuition.estimate_cost(operator, {})
        
        self.actr.set_llm_estimators(prob_estimator, cost_estimator)
    
    def solve(
        self,
        problem: Dict[str, Any],
        goal: Dict[str, Any],
        operators: Optional[List[SoarOperator]] = None
    ) -> ReasoningResult:
        """
        Main entry point for solving a problem.
        
        Args:
            problem: Problem description
            goal: Goal specification
            operators: Available operators (optional)
            
        Returns:
            ReasoningResult with solution and trace
        """
        session_id = str(uuid.uuid4())
        logger.info(f"Starting reasoning session {session_id}")
        
        # Create session
        session = ReasoningSession(
            session_id=session_id,
            problem=problem,
            goal=goal,
            orchestrator=self.orchestrator,
            config=self.config
        )
        
        session.start()
        self.session_count += 1
        
        # Initialize Soar
        default_operators = operators or self._default_operators()
        self.soar.initialize(goal, default_operators)
        
        # Main reasoning loop
        solution = None
        impasse_to_learn = None
        
        try:
            start_time = time.time()
            max_time = self.config.max_reasoning_time_ms / 1000
            
            while (time.time() - start_time) < max_time:
                # Check which system to use
                current_system = session.check_system_switch()
                
                if current_system == SystemType.SOAR:
                    # Run Soar cycle
                    success, impasse = session.run_soar_cycle()
                    
                    if success:
                        # Check if goal achieved
                        state = self.soar.get_current_state()
                        if self._goal_achieved(state, goal):
                            solution = state
                            break
                    
                    if impasse:
                        impasse_to_learn = impasse
                        # Try to handle impasse via subgoal
                        resolution = session.handle_impasse(impasse)
                        
                        if not resolution["subgoal_created"]:
                            # Cannot create subgoal, pressure will rise
                            pass
                
                elif current_system == SystemType.ACT_R:
                    # Run ACT-R cycle
                    selected = session.run_actr_cycle()
                    
                    if selected:
                        # Check if successful
                        if self._actr_success(selected, goal):
                            solution = {"operator": selected}
                            break
                
                elif current_system == SystemType.EVOLUTIONARY:
                    # Run evolutionary fallback
                    best = session.run_evolutionary_fallback()
                    
                    if best and best.fitness > 0.8:
                        solution = best
                        break
            
            # Learn from successful resolution
            if solution and impasse_to_learn:
                resolution = {"value": solution}
                session.learn_from_success(impasse_to_learn, resolution)
            
            # Finalize
            success = solution is not None
            if success:
                self.success_count += 1
            
            session.finalize(success, solution)
            
        except Exception as e:
            logger.error(f"Error in reasoning session: {e}", exc_info=True)
            session.finalize(False, None)
        
        return session.result
    
    def _default_operators(self) -> List[SoarOperator]:
        """Get default operators."""
        # Return some basic operators
        return [
            SoarOperator(
                name="explore",
                preconditions=[],
                actions=[{"type": "add", "attribute": "status", "value": "exploring"}],
                preferences={"default": 0.5}
            ),
            SoarOperator(
                name="evaluate",
                preconditions=[{"attribute": "status", "value": "exploring"}],
                actions=[{"type": "add", "attribute": "status", "value": "evaluating"}],
                preferences={"default": 0.6}
            ),
        ]
    
    def _goal_achieved(self, state: SoarState, goal: Dict[str, Any]) -> bool:
        """Check if goal is achieved."""
        if not state:
            return False
        
        for key, value in goal.items():
            state_value = state.get_wme_attribute(key)
            if state_value != value:
                return False
        
        return True
    
    def _actr_success(self, operator: ACTRProduction, goal: Dict[str, Any]) -> bool:
        """Check if ACT-R operator achieves goal."""
        # Simplified check
        return operator.utility > 0.8
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "session_count": self.session_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.session_count if self.session_count > 0 else 0.0,
            "soar_stats": self.soar.get_stats(),
            "actr_stats": self.actr.get_stats(),
            "pressure_stats": self.pressure_valve.get_stats(),
            "llm_stats": self.llm_intuition.get_stats(),
            "evolutionary_stats": self.evolutionary.get_stats(),
            "chunking_stats": self.chunking.get_stats()
        }

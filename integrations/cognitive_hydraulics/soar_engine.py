"""Soar Cognitive Architecture Implementation.

System 2: Slow, deliberate, symbolic reasoning.
Rule-based pattern matching with impasse detection.

Architecture:
    - SoarWorkingMemory: Working memory for state/context (limited slots)
    - SoarProductionSystem: Rule matching and firing
    - SoarDecisionCycle: Elaboration -> Proposal -> Selection -> Application
    - ImpasseDetector: Detects tie impasses, no-change impasses
    - SubgoalManager: Creates and manages subgoals
    - ChunkingSystem: Learns from successful resolutions

Key Concepts:
    - Impasse: When reasoning cannot proceed (tie, no-change, conflict)
    - Subgoal: Temporary goal created to resolve an impasse
    - Chunk: Learned rule from successful subgoal resolution
    - Operators: Actions that transform the current state
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timezone
from enum import Enum, auto
from copy import deepcopy

from .config import SoarConfig

logger = logging.getLogger(__name__)


class ImpasseType(Enum):
    """Types of impasses in Soar."""
    TIE = auto()  # Multiple operators with equal preferences
    NO_CHANGE = auto()  # No operator proposed
    CONFLICT = auto()  # Operators with conflicting preferences
    CONSTRAINT_FAILURE = auto()  # Constraints cannot be satisfied


@dataclass
class Impasse:
    """Base class for impasses."""
    impasse_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    impasse_type: ImpasseType = ImpasseType.NO_CHANGE
    state_id: str = ""
    description: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "impasse_id": self.impasse_id,
            "impasse_type": self.impasse_type.name,
            "state_id": self.state_id,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class TieImpasse(Impasse):
    """Multiple operators with equal preferences."""
    tied_operators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.impasse_type = ImpasseType.TIE
        self.description = f"Tie between operators: {self.tied_operators}"


@dataclass
class NoChangeImpasse(Impasse):
    """No operator proposed for current state."""
    last_operator: Optional[str] = None
    
    def __post_init__(self):
        self.impasse_type = ImpasseType.NO_CHANGE
        self.description = f"No operator proposed (last: {self.last_operator})"


@dataclass
class ConflictImpasse(Impasse):
    """Operators with conflicting preferences."""
    conflicting_operators: List[Tuple[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        self.impasse_type = ImpasseType.CONFLICT
        self.description = f"Conflicting operators: {self.conflicting_operators}"


@dataclass
class ConstraintFailureImpasse(Impasse):
    """Constraints cannot be satisfied."""
    failed_constraints: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.impasse_type = ImpasseType.CONSTRAINT_FAILURE
        self.description = f"Failed constraints: {self.failed_constraints}"


@dataclass
class SoarOperator:
    """An operator that can transform a state."""
    operator_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    preconditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, float] = field(default_factory=dict)
    
    def is_applicable(self, state: "SoarState") -> bool:
        """Check if operator is applicable to state."""
        for precond in self.preconditions:
            if not self._check_precondition(precond, state):
                return False
        return True
    
    def _check_precondition(self, precond: Dict[str, Any], state: "SoarState") -> bool:
        """Check a single precondition against state."""
        attr = precond.get("attribute")
        value = precond.get("value")
        op = precond.get("operator", "equals")
        
        state_value = state.get_wme_attribute(attr)
        
        if op == "equals":
            return state_value == value
        elif op == "exists":
            return state_value is not None
        elif op == "not_equals":
            return state_value != value
        elif op == "in":
            return state_value in value if isinstance(value, list) else False
        
        return False
    
    def apply(self, state: "SoarState") -> "SoarState":
        """Apply operator to state, return new state."""
        new_state = deepcopy(state)
        
        for action in self.actions:
            action_type = action.get("type")
            if action_type == "add":
                new_state.add_wme(action.get("attribute"), action.get("value"))
            elif action_type == "remove":
                new_state.remove_wme(action.get("attribute"))
            elif action_type == "modify":
                new_state.modify_wme(action.get("attribute"), action.get("value"))
        
        new_state.operator_proposed = self.operator_id
        return new_state


@dataclass
class SoarState:
    """Represents current problem state in Soar."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: Dict[str, Any] = field(default_factory=dict)
    working_memory_elements: Dict[str, Any] = field(default_factory=dict)
    subgoal_depth: int = 0
    parent_state_id: Optional[str] = None
    operator_proposed: Optional[str] = None
    operator_selected: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Track changes for elaboration
    elaborated_attributes: Set[str] = field(default_factory=set)
    
    def add_wme(self, attribute: str, value: Any):
        """Add working memory element."""
        self.working_memory_elements[attribute] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def remove_wme(self, attribute: str):
        """Remove working memory element."""
        if attribute in self.working_memory_elements:
            del self.working_memory_elements[attribute]
    
    def modify_wme(self, attribute: str, value: Any):
        """Modify working memory element."""
        self.working_memory_elements[attribute] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_wme_attribute(self, attribute: str) -> Any:
        """Get value of a working memory element."""
        wme = self.working_memory_elements.get(attribute)
        return wme["value"] if wme else None
    
    def get_all_attributes(self) -> Dict[str, Any]:
        """Get all working memory elements."""
        return {k: v["value"] for k, v in self.working_memory_elements.items()}


@dataclass
class SoarRule:
    """Production rule (IF-THEN)."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    learned: bool = False  # True if this is a learned chunk
    utility: float = 0.0
    creation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def matches(self, state: SoarState) -> bool:
        """Check if rule matches state."""
        for condition in self.conditions:
            if not self._evaluate_condition(condition, state):
                return False
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], state: SoarState) -> bool:
        """Evaluate a single condition."""
        attr = condition.get("attribute")
        expected = condition.get("value")
        op = condition.get("operator", "equals")
        
        actual = state.get_wme_attribute(attr)
        
        if op == "equals":
            return actual == expected
        elif op == "not_equals":
            return actual != expected
        elif op == "exists":
            return actual is not None
        elif op == "greater":
            return actual is not None and actual > expected
        elif op == "less":
            return actual is not None and actual < expected
        
        return False
    
    def fire(self, state: SoarState) -> List[Dict[str, Any]]:
        """Fire rule, return actions to apply."""
        logger.debug(f"Firing rule: {self.name}")
        return self.actions


class SoarWorkingMemory:
    """Working memory management for Soar."""
    
    def __init__(self, config: SoarConfig):
        self.config = config
        self.states: Dict[str, SoarState] = {}
        self.current_state_id: Optional[str] = None
        self.wm_change_log: List[Dict] = []
    
    def create_state(self, goal: Dict[str, Any], parent_id: Optional[str] = None) -> SoarState:
        """Create a new state."""
        depth = 0
        if parent_id and parent_id in self.states:
            depth = self.states[parent_id].subgoal_depth + 1
        
        state = SoarState(
            goal=goal,
            subgoal_depth=depth,
            parent_state_id=parent_id
        )
        
        self.states[state.state_id] = state
        
        if self.current_state_id is None:
            self.current_state_id = state.state_id
        
        logger.debug(f"Created state {state.state_id} at depth {depth}")
        return state
    
    def get_current_state(self) -> Optional[SoarState]:
        """Get the current state."""
        if self.current_state_id:
            return self.states.get(self.current_state_id)
        return None
    
    def switch_state(self, state_id: str):
        """Switch to a different state."""
        if state_id in self.states:
            old_state = self.current_state_id
            self.current_state_id = state_id
            self.wm_change_log.append({
                "type": "state_switch",
                "from": old_state,
                "to": state_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    def add_to_current(self, attribute: str, value: Any):
        """Add WME to current state."""
        state = self.get_current_state()
        if state:
            state.add_wme(attribute, value)
            self.wm_change_log.append({
                "type": "add",
                "attribute": attribute,
                "state": state.state_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })


class SoarProductionSystem:
    """Production rule system for Soar."""
    
    def __init__(self, config: SoarConfig):
        self.config = config
        self.rules: Dict[str, SoarRule] = {}
        self.firing_count: Dict[str, int] = {}
    
    def add_rule(self, rule: SoarRule):
        """Add a production rule."""
        self.rules[rule.rule_id] = rule
        self.firing_count[rule.rule_id] = 0
        logger.debug(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove a production rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            del self.firing_count[rule_id]
    
    def match_rules(self, state: SoarState) -> List[SoarRule]:
        """Find all matching rules for a state."""
        matches = []
        for rule in self.rules.values():
            if rule.matches(state):
                matches.append(rule)
        return matches
    
    def fire_rules(self, state: SoarState) -> List[Dict[str, Any]]:
        """Fire all matching rules, collect actions."""
        all_actions = []
        matches = self.match_rules(state)
        
        for rule in matches:
            actions = rule.fire(state)
            all_actions.extend(actions)
            self.firing_count[rule.rule_id] += 1
        
        return all_actions


class ImpasseDetector:
    """Detects impasses in Soar reasoning."""
    
    def __init__(self, config: SoarConfig):
        self.config = config
        self.impasse_history: List[Impasse] = []
        self.impasse_count = 0
    
    def detect_impasse(
        self,
        state: SoarState,
        proposed_operators: List[SoarOperator],
        cycle_count: int
    ) -> Optional[Impasse]:
        """Detect if there's an impasse in the current state."""
        
        # No operators proposed
        if not proposed_operators:
            impasse = NoChangeImpasse(
                state_id=state.state_id,
                last_operator=state.operator_selected
            )
            self._record_impasse(impasse)
            return impasse
        
        # Multiple operators with no clear winner (tie)
        if len(proposed_operators) >= self.config.tie_impasse_threshold:
            # Check if they have equal preferences
            best_preference = max(
                sum(op.preferences.values()) for op in proposed_operators
            )
            tied = [
                op.operator_id for op in proposed_operators
                if abs(sum(op.preferences.values()) - best_preference) < 0.01
            ]
            
            if len(tied) >= self.config.tie_impasse_threshold:
                impasse = TieImpasse(
                    state_id=state.state_id,
                    tied_operators=tied
                )
                self._record_impasse(impasse)
                return impasse
        
        return None
    
    def _record_impasse(self, impasse: Impasse):
        """Record an impasse in history."""
        self.impasse_history.append(impasse)
        self.impasse_count += 1
        logger.info(f"Detected {impasse.impasse_type.name} impasse: {impasse.description}")


class SubgoalManager:
    """Manages subgoals for impasse resolution."""
    
    def __init__(self, config: SoarConfig, working_memory: SoarWorkingMemory):
        self.config = config
        self.working_memory = working_memory
        self.active_subgoals: Dict[str, Dict] = {}
        self.subgoal_stack: List[str] = []
    
    def create_subgoal(
        self,
        impasse: Impasse,
        parent_state: SoarState
    ) -> SoarState:
        """Create a subgoal to resolve an impasse."""
        if parent_state.subgoal_depth >= self.config.max_subgoal_depth:
            logger.warning("Max subgoal depth reached, cannot create subgoal")
            return parent_state
        
        # Create subgoal state with impasse as goal
        subgoal_state = self.working_memory.create_state(
            goal={
                "type": "resolve_impasse",
                "impasse_type": impasse.impasse_type.name,
                "impasse_id": impasse.impasse_id,
                "parent_state_id": parent_state.state_id,
            },
            parent_id=parent_state.state_id
        )
        
        # Add impasse context
        subgoal_state.add_wme("impasse_type", impasse.impasse_type.name)
        subgoal_state.add_wme("parent_context", impasse.context)
        
        # Track active subgoal
        self.active_subgoals[subgoal_state.state_id] = {
            "impasse": impasse,
            "parent_state_id": parent_state.state_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "resolved": False
        }
        
        self.subgoal_stack.append(subgoal_state.state_id)
        
        # Switch to subgoal
        self.working_memory.switch_state(subgoal_state.state_id)
        
        logger.info(f"Created subgoal {subgoal_state.state_id} for {impasse.impasse_type.name}")
        return subgoal_state
    
    def resolve_subgoal(self, state_id: str, result: Dict[str, Any]) -> Optional[str]:
        """Resolve a subgoal and return to parent."""
        if state_id not in self.active_subgoals:
            return None
        
        subgoal_info = self.active_subgoals[state_id]
        subgoal_info["resolved"] = True
        subgoal_info["result"] = result
        
        parent_id = subgoal_info["parent_state_id"]
        
        # Remove from stack
        if state_id in self.subgoal_stack:
            self.subgoal_stack.remove(state_id)
        
        # Switch back to parent
        self.working_memory.switch_state(parent_id)
        
        logger.info(f"Resolved subgoal {state_id}, returning to {parent_id}")
        return parent_id
    
    def get_current_depth(self) -> int:
        """Get current subgoal depth."""
        return len(self.subgoal_stack)


class ChunkingSystem:
    """Learn chunks (rules) from successful impasse resolutions."""
    
    def __init__(self, config: SoarConfig, production_system: SoarProductionSystem):
        self.config = config
        self.production_system = production_system
        self.chunks: Dict[str, SoarRule] = {}
        self.resolution_history: List[Dict] = []
    
    def create_chunk(
        self,
        impasse: Impasse,
        resolution: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[SoarRule]:
        """Create a chunk from a successful resolution."""
        if not self.config.enable_chunking:
            return None
        
        # Extract conditions from impasse context
        conditions = self._extract_conditions(impasse, context)
        
        # Extract actions from resolution
        actions = self._extract_actions(resolution)
        
        # Create rule
        chunk = SoarRule(
            name=f"chunk_{impasse.impasse_type.name.lower()}_{uuid.uuid4().hex[:8]}",
            conditions=conditions,
            actions=actions,
            learned=True,
            utility=0.5
        )
        
        self.chunks[chunk.rule_id] = chunk
        self.production_system.add_rule(chunk)
        
        self.resolution_history.append({
            "impasse_type": impasse.impasse_type.name,
            "chunk_id": chunk.rule_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        logger.info(f"Created chunk: {chunk.name}")
        return chunk
    
    def _extract_conditions(
        self,
        impasse: Impasse,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract conditions from impasse context."""
        conditions = []
        
        # Add impasse type as condition
        conditions.append({
            "attribute": "impasse_type",
            "operator": "equals",
            "value": impasse.impasse_type.name
        })
        
        # Add context conditions
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append({
                    "attribute": key,
                    "operator": "equals",
                    "value": value
                })
        
        return conditions
    
    def _extract_actions(self, resolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actions from resolution."""
        actions = []
        
        if "operator_id" in resolution:
            actions.append({
                "type": "propose",
                "operator_id": resolution["operator_id"]
            })
        
        if "value" in resolution:
            actions.append({
                "type": "add",
                "attribute": "resolution_value",
                "value": resolution["value"]
            })
        
        return actions
    
    def generalize_chunk(self, chunk: SoarRule) -> SoarRule:
        """Make a chunk more general by removing specific values."""
        # This is a simplified generalization
        generalized_conditions = []
        
        for cond in chunk.conditions:
            # Keep structure but may generalize specific values
            generalized = cond.copy()
            
            # Generalize UUIDs and timestamps
            if isinstance(cond.get("value"), str):
                if len(cond["value"]) == 36 and "-" in cond["value"]:
                    # Likely a UUID - remove this condition
                    continue
            
            generalized_conditions.append(generalized)
        
        generalized = SoarRule(
            name=f"generalized_{chunk.name}",
            conditions=generalized_conditions,
            actions=chunk.actions,
            learned=True,
            utility=chunk.utility * 0.9  # Slightly lower utility for generalized
        )
        
        return generalized


class SoarDecisionCycle:
    """Soar Decision Cycle: Elaboration -> Proposal -> Selection -> Application."""
    
    def __init__(
        self,
        config: SoarConfig,
        working_memory: SoarWorkingMemory,
        production_system: SoarProductionSystem,
        impasse_detector: ImpasseDetector,
        subgoal_manager: SubgoalManager,
        chunking_system: ChunkingSystem
    ):
        self.config = config
        self.working_memory = working_memory
        self.production_system = production_system
        self.impasse_detector = impasse_detector
        self.subgoal_manager = subgoal_manager
        self.chunking_system = chunking_system
        
        self.cycle_count = 0
        self.decision_log: List[Dict] = []
    
    def run_cycle(self, available_operators: List[SoarOperator]) -> Tuple[bool, Optional[Impasse]]:
        """
        Run one Soar decision cycle.
        
        Returns:
            (success, impasse): success=True if cycle completed, impasse if detected
        """
        state = self.working_memory.get_current_state()
        if not state:
            return False, None
        
        self.cycle_count += 1
        
        if self.cycle_count > self.config.max_decision_cycles:
            logger.warning("Max decision cycles reached")
            return False, None
        
        logger.debug(f"Decision cycle {self.cycle_count}")
        
        # Phase 1: Elaboration - Fire production rules
        self._elaboration_phase(state)
        
        # Phase 2: Proposal - Find applicable operators
        proposed = self._proposal_phase(state, available_operators)
        
        # Phase 3: Check for impasse
        impasse = self.impasse_detector.detect_impasse(state, proposed, self.cycle_count)
        if impasse:
            return False, impasse
        
        # Phase 4: Selection - Choose best operator
        selected = self._selection_phase(proposed)
        if not selected:
            return False, NoChangeImpasse(state_id=state.state_id)
        
        # Phase 5: Application - Apply selected operator
        self._application_phase(state, selected)
        
        return True, None
    
    def _elaboration_phase(self, state: SoarState):
        """Fire production rules to elaborate state."""
        actions = self.production_system.fire_rules(state)
        
        for action in actions:
            if action.get("type") == "add":
                state.add_wme(action.get("attribute"), action.get("value"))
    
    def _proposal_phase(
        self,
        state: SoarState,
        operators: List[SoarOperator]
    ) -> List[SoarOperator]:
        """Find applicable operators."""
        proposed = []
        for op in operators:
            if op.is_applicable(state):
                proposed.append(op)
        
        return proposed
    
    def _selection_phase(self, proposed: List[SoarOperator]) -> Optional[SoarOperator]:
        """Select best operator based on preferences."""
        if not proposed:
            return None
        
        # Calculate total preference for each operator
        def get_preference(op: SoarOperator) -> float:
            return sum(op.preferences.values())
        
        # Sort by preference (descending)
        proposed.sort(key=get_preference, reverse=True)
        
        return proposed[0]
    
    def _application_phase(self, state: SoarState, operator: SoarOperator):
        """Apply selected operator."""
        new_state = operator.apply(state)
        
        # Update state reference
        self.working_memory.states[new_state.state_id] = new_state
        self.working_memory.switch_state(new_state.state_id)
        
        self.decision_log.append({
            "cycle": self.cycle_count,
            "operator_id": operator.operator_id,
            "operator_name": operator.name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


class SoarEngine:
    """Main Soar Engine - System 2 Symbolic Reasoning."""
    
    def __init__(self, config: Optional[SoarConfig] = None):
        self.config = config or SoarConfig()
        
        # Core components
        self.working_memory = SoarWorkingMemory(self.config)
        self.production_system = SoarProductionSystem(self.config)
        self.impasse_detector = ImpasseDetector(self.config)
        self.subgoal_manager = SubgoalManager(self.config, self.working_memory)
        self.chunking_system = ChunkingSystem(self.config, self.production_system)
        
        # Decision cycle
        self.decision_cycle = SoarDecisionCycle(
            self.config,
            self.working_memory,
            self.production_system,
            self.impasse_detector,
            self.subgoal_manager,
            self.chunking_system
        )
        
        # State
        self.operators: Dict[str, SoarOperator] = {}
        self.initialized = False
    
    def initialize(self, goal: Dict[str, Any], operators: List[SoarOperator]):
        """Initialize the engine with a goal and operators."""
        # Store operators
        for op in operators:
            self.operators[op.operator_id] = op
        
        # Create initial state
        self.working_memory.create_state(goal=goal)
        
        self.initialized = True
        logger.info("SoarEngine initialized")
    
    def add_rule(self, rule: SoarRule):
        """Add a production rule."""
        self.production_system.add_rule(rule)
    
    def run_decision_cycle(self) -> Tuple[bool, Optional[Impasse]]:
        """Execute one Soar decision cycle."""
        if not self.initialized:
            raise RuntimeError("Engine not initialized")
        
        return self.decision_cycle.run_cycle(list(self.operators.values()))
    
    def detect_impasse(self, state: SoarState, proposed: List[SoarOperator]) -> Optional[Impasse]:
        """Check for reasoning blocks."""
        return self.impasse_detector.detect_impasse(
            state, proposed, self.decision_cycle.cycle_count
        )
    
    def create_subgoal(self, impasse: Impasse) -> SoarState:
        """Create subgoal to resolve impasse."""
        state = self.working_memory.get_current_state()
        if state:
            return self.subgoal_manager.create_subgoal(impasse, state)
        raise RuntimeError("No current state")
    
    def chunk_success(self, impasse: Impasse, resolution: Dict[str, Any], context: Dict[str, Any]):
        """Learn rule from successful resolution."""
        return self.chunking_system.create_chunk(impasse, resolution, context)
    
    def get_current_state(self) -> Optional[SoarState]:
        """Get the current working state."""
        return self.working_memory.get_current_state()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "cycle_count": self.decision_cycle.cycle_count,
            "impasse_count": self.impasse_detector.impasse_count,
            "subgoal_depth": self.subgoal_manager.get_current_depth(),
            "chunks_learned": len(self.chunking_system.chunks),
            "rules_count": len(self.production_system.rules),
            "operators_count": len(self.operators)
        }

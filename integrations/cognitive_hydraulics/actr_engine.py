"""ACT-R Cognitive Architecture Implementation.

System 1: Fast, heuristic, utility-based reasoning.

Utility Equation:
    U = P × G - C - HistoryPenalty + Noise(s)
    
    where:
        P = Probability of success (from LLM)
        G = Goal value (importance)
        C = Cost (time/effort from LLM)
        HistoryPenalty = Tabu penalty for recently used operators
        Noise = Stochastic noise ~ N(0, σ)

Activation (ACT-R):
    A = ln(Σ t_i^(-d)) + noise
    where t_i = time since i-th presentation, d = decay parameter

Architecture:
    - ACTRDeclarativeMemory: Fact storage with activation
    - ACTRProceduralMemory: Production rules with utilities
    - UtilityCalculator: U = P × G - C + Noise
    - TabuSearch: Prevents operator loops
    - NoiseGenerator: Stochastic noise for variability
"""

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timezone, timedelta
from collections import deque

import numpy as np

from .config import ACTRConfig

logger = logging.getLogger(__name__)


@dataclass
class ACTRChunk:
    """Declarative memory chunk."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_type: str = ""
    slots: Dict[str, Any] = field(default_factory=dict)
    
    # Activation tracking
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    presentations: List[datetime] = field(default_factory=list)
    base_activation: float = 0.0
    
    # Contextual activation
    context_strength: float = 0.0
    
    def __post_init__(self):
        """Initialize presentations list."""
        if not self.presentations:
            self.presentations = [self.creation_time]
    
    def get_activation(self, decay: float, noise_sigma: float) -> float:
        """
        Calculate activation using ACT-R formula:
        A = ln(Σ t_i^(-d)) + noise
        """
        now = datetime.now(timezone.utc)
        
        # Calculate base level activation from presentations
        sum_term = 0.0
        for t_i in self.presentations:
            time_diff = (now - t_i).total_seconds()
            if time_diff > 0:
                sum_term += time_diff ** (-decay)
        
        if sum_term > 0:
            self.base_activation = math.log(sum_term)
        else:
            self.base_activation = -10.0  # Very low activation
        
        # Add stochastic noise
        noise = random.gauss(0, noise_sigma)
        
        # Total activation includes base + contextual + noise
        total = self.base_activation + self.context_strength + noise
        
        return total
    
    def present(self):
        """Record a presentation of this chunk."""
        self.presentations.append(datetime.now(timezone.utc))
    
    def matches(self, pattern: Dict[str, Any]) -> bool:
        """Check if chunk matches a pattern."""
        for slot, value in pattern.items():
            if self.slots.get(slot) != value:
                return False
        return True


@dataclass
class ACTRProduction:
    """Procedural rule with utility."""
    production_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Utility parameters
    utility: float = 0.0
    probability: float = 0.5  # P in U = P × G - C
    cost: float = 1.0  # C in U = P × G - C
    goal_value: float = 10.0  # G in U = P × G - C
    
    # Learning
    success_count: int = 0
    failure_count: int = 0
    
    def calculate_utility(
        self,
        history_penalty: float = 0.0,
        noise_sigma: float = 0.0
    ) -> float:
        """
        Calculate utility using ACT-R equation:
        U = P × G - C - HistoryPenalty + Noise(s)
        """
        noise = random.gauss(0, noise_sigma)
        
        u = (self.probability * self.goal_value) - self.cost - history_penalty + noise
        
        return u
    
    def update_from_outcome(self, success: bool):
        """Update utility from outcome."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        total = self.success_count + self.failure_count
        if total > 0:
            # Update probability estimate
            self.probability = self.success_count / total


@dataclass
class TabuList:
    """Recent operators to avoid (prevents loops)."""
    max_size: int = 10
    entries: deque = field(default_factory=lambda: deque(maxlen=10))
    penalty_base: float = 1.0
    
    def __post_init__(self):
        """Ensure deque has correct maxlen."""
        if self.entries.maxlen != self.max_size:
            self.entries = deque(self.entries, maxlen=self.max_size)
    
    def add(self, operator_id: str):
        """Add operator to tabu list."""
        self.entries.append({
            "operator_id": operator_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def contains(self, operator_id: str) -> bool:
        """Check if operator is in tabu list."""
        return any(entry["operator_id"] == operator_id for entry in self.entries)
    
    def get_penalty(self, operator_id: str) -> float:
        """Get penalty for operator based on recency in tabu list."""
        for i, entry in enumerate(self.entries):
            if entry["operator_id"] == operator_id:
                # More recent = higher penalty
                recency_factor = 1.0 - (i / len(self.entries))
                return self.penalty_base * recency_factor
        return 0.0
    
    def clear(self):
        """Clear the tabu list."""
        self.entries.clear()


class UtilityEquation:
    """U = P × G - C - HistoryPenalty + Noise(s)"""
    
    def __init__(self, config: ACTRConfig):
        self.config = config
    
    def compute(
        self,
        probability: float,
        goal_value: float,
        cost: float,
        history_penalty: float = 0.0
    ) -> float:
        """
        Compute utility using ACT-R equation.
        
        Args:
            probability: P(success) from 0 to 1
            goal_value: Importance of goal
            cost: Time/effort estimate
            history_penalty: Tabu penalty
            
        Returns:
            Utility value (higher is better)
        """
        noise = random.gauss(0, self.config.noise_sigma)
        
        utility = (probability * goal_value) - cost - history_penalty + noise
        
        return utility
    
    def compute_batch(
        self,
        operators: List[ACTRProduction],
        tabu_list: TabuList
    ) -> List[Tuple[ACTRProduction, float]]:
        """Compute utility for multiple operators."""
        results = []
        
        for op in operators:
            history_penalty = tabu_list.get_penalty(op.production_id)
            utility = self.compute(
                op.probability,
                op.goal_value,
                op.cost,
                history_penalty
            )
            results.append((op, utility))
        
        return results


class NoiseGenerator:
    """Stochastic noise for variability."""
    
    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma
    
    def generate(self) -> float:
        """Generate noise ~ N(0, σ)."""
        return random.gauss(0, self.sigma)
    
    def generate_batch(self, n: int) -> List[float]:
        """Generate batch of noise values."""
        return [self.generate() for _ in range(n)]


class ACTRDeclarativeMemory:
    """Fact storage with activation-based retrieval."""
    
    def __init__(self, config: ACTRConfig):
        self.config = config
        self.chunks: Dict[str, ACTRChunk] = {}
        self.type_index: Dict[str, List[str]] = {}
    
    def add_chunk(self, chunk: ACTRChunk) -> str:
        """Add a chunk to declarative memory."""
        self.chunks[chunk.chunk_id] = chunk
        
        # Index by type
        if chunk.chunk_type not in self.type_index:
            self.type_index[chunk.chunk_type] = []
        self.type_index[chunk.chunk_type].append(chunk.chunk_id)
        
        logger.debug(f"Added chunk {chunk.chunk_id} of type {chunk.chunk_type}")
        return chunk.chunk_id
    
    def retrieve_chunk(self, chunk_id: str) -> Optional[ACTRChunk]:
        """Retrieve a chunk by ID."""
        chunk = self.chunks.get(chunk_id)
        if chunk:
            chunk.present()
        return chunk
    
    def retrieve_by_pattern(
        self,
        pattern: Dict[str, Any],
        chunk_type: Optional[str] = None
    ) -> Optional[ACTRChunk]:
        """
        Retrieve best matching chunk by pattern.
        Uses activation to select among matches.
        """
        candidates = []
        
        # Get candidate IDs
        if chunk_type and chunk_type in self.type_index:
            candidate_ids = self.type_index[chunk_type]
        else:
            candidate_ids = list(self.chunks.keys())
        
        # Find matches and calculate activations
        for cid in candidate_ids:
            chunk = self.chunks[cid]
            if chunk.matches(pattern):
                activation = chunk.get_activation(
                    self.config.activation_decay,
                    self.config.noise_sigma
                )
                candidates.append((chunk, activation))
        
        if not candidates:
            return None
        
        # Select highest activation
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_chunk = candidates[0][0]
        best_chunk.present()
        
        return best_chunk
    
    def partial_match_retrieval(
        self,
        pattern: Dict[str, Any],
        similarity_threshold: float = 0.5
    ) -> Optional[ACTRChunk]:
        """Retrieve chunk using partial matching."""
        best_chunk = None
        best_score = -float('inf')
        
        for chunk in self.chunks.values():
            # Calculate match score
            match_count = sum(
                1 for slot, value in pattern.items()
                if chunk.slots.get(slot) == value
            )
            total_slots = len(pattern)
            
            if total_slots > 0:
                match_score = match_count / total_slots
            else:
                match_score = 0.0
            
            if match_score >= similarity_threshold:
                activation = chunk.get_activation(
                    self.config.activation_decay,
                    self.config.noise_sigma
                )
                
                # Combined score: activation + match quality
                score = activation + match_score
                
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
        
        if best_chunk:
            best_chunk.present()
        
        return best_chunk
    
    def get_all_chunks(self) -> List[ACTRChunk]:
        """Get all chunks."""
        return list(self.chunks.values())


class ACTRProceduralMemory:
    """Production rules with utility."""
    
    def __init__(self, config: ACTRConfig):
        self.config = config
        self.productions: Dict[str, ACTRProduction] = {}
    
    def add_production(self, production: ACTRProduction) -> str:
        """Add a production rule."""
        self.productions[production.production_id] = production
        logger.debug(f"Added production: {production.name}")
        return production.production_id
    
    def get_production(self, production_id: str) -> Optional[ACTRProduction]:
        """Get a production by ID."""
        return self.productions.get(production_id)
    
    def find_matching_productions(
        self,
        context: Dict[str, Any]
    ) -> List[ACTRProduction]:
        """Find productions that match the current context."""
        matches = []
        
        for production in self.productions.values():
            if self._matches_context(production, context):
                matches.append(production)
        
        return matches
    
    def _matches_context(
        self,
        production: ACTRProduction,
        context: Dict[str, Any]
    ) -> bool:
        """Check if production matches context."""
        for condition in production.conditions:
            slot = condition.get("slot")
            value = condition.get("value")
            op = condition.get("operator", "equals")
            
            context_value = context.get(slot)
            
            if op == "equals":
                if context_value != value:
                    return False
            elif op == "exists":
                if context_value is None:
                    return False
            elif op == "greater":
                if context_value is None or context_value <= value:
                    return False
            elif op == "less":
                if context_value is None or context_value >= value:
                    return False
        
        return True
    
    def update_utility(self, production_id: str, success: bool):
        """Update utility based on outcome."""
        production = self.productions.get(production_id)
        if production:
            production.update_from_outcome(success)


class UtilityCalculator:
    """Calculate utilities for operator selection."""
    
    def __init__(self, config: ACTRConfig):
        self.config = config
        self.equation = UtilityEquation(config)
        self.tabu = TabuList(
            max_size=config.tabu_list_size,
            penalty_base=config.tabu_penalty_weight
        )
        self.noise = NoiseGenerator(config.noise_sigma)
    
    def compute_utility(
        self,
        operator: ACTRProduction,
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate U = P × G - C - HistoryPenalty + Noise(s)
        
        Args:
            operator: The operator to evaluate
            goal: Current goal
            context: Current context
            
        Returns:
            Utility value
        """
        history_penalty = self.tabu.get_penalty(operator.production_id)
        
        utility = self.equation.compute(
            operator.probability,
            operator.goal_value,
            operator.cost,
            history_penalty
        )
        
        return utility
    
    def select_operator(
        self,
        operators: List[ACTRProduction],
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[ACTRProduction]:
        """
        Select operator with highest utility.
        
        Args:
            operators: Available operators
            goal: Current goal
            context: Current context
            
        Returns:
            Selected operator or None
        """
        if not operators:
            return None
        
        # Calculate utilities
        utilities = []
        for op in operators:
            u = self.compute_utility(op, goal, context)
            utilities.append((op, u))
        
        # Sort by utility (descending)
        utilities.sort(key=lambda x: x[1], reverse=True)
        
        # Select best
        selected = utilities[0][0]
        
        # Update tabu list
        self.tabu.add(selected.production_id)
        
        logger.debug(f"Selected operator {selected.name} with utility {utilities[0][1]:.2f}")
        
        return selected
    
    def update_history(self, operator_id: str):
        """Add operator to tabu list."""
        self.tabu.add(operator_id)
    
    def estimate_probability(
        self,
        operator: ACTRProduction,
        goal: Dict[str, Any],
        llm_estimator: Optional[Callable] = None
    ) -> float:
        """
        Estimate P(success) for operator.
        
        Uses LLM if available, otherwise uses stored probability.
        """
        if llm_estimator:
            try:
                estimated = llm_estimator(operator, goal)
                if 0 <= estimated <= 1:
                    return estimated
            except Exception as e:
                logger.warning(f"LLM probability estimation failed: {e}")
        
        return operator.probability
    
    def estimate_cost(
        self,
        operator: ACTRProduction,
        llm_estimator: Optional[Callable] = None
    ) -> float:
        """
        Estimate C(time/effort) for operator.
        
        Uses LLM if available, otherwise uses stored cost.
        """
        if llm_estimator:
            try:
                estimated = llm_estimator(operator)
                if estimated >= 0:
                    return estimated
            except Exception as e:
                logger.warning(f"LLM cost estimation failed: {e}")
        
        return operator.cost


class TabuSearch:
    """Tabu search with history penalty."""
    
    def __init__(self, config: ACTRConfig):
        self.tabu_list = TabuList(
            max_size=config.tabu_list_size,
            penalty_base=config.tabu_penalty_weight
        )
        self.history_penalty_base = config.history_penalty_base
    
    def add_to_history(self, operator_id: str):
        """Add operator to history."""
        self.tabu_list.add(operator_id)
    
    def get_penalty(self, operator_id: str) -> float:
        """Get history penalty for operator."""
        return self.tabu_list.get_penalty(operator_id)
    
    def is_tabu(self, operator_id: str) -> bool:
        """Check if operator is tabu."""
        return self.tabu_list.contains(operator_id)
    
    def clear_history(self):
        """Clear tabu history."""
        self.tabu_list.clear()
    
    def get_recent_operators(self, n: int = 5) -> List[str]:
        """Get n most recent operators."""
        recent = []
        for entry in self.tabu_list.entries:
            recent.append(entry["operator_id"])
            if len(recent) >= n:
                break
        return recent


class ACTREngine:
    """Main ACT-R Engine - System 1 Heuristic Reasoning."""
    
    def __init__(self, config: Optional[ACTRConfig] = None):
        self.config = config or ACTRConfig()
        
        # Core components
        self.declarative_memory = ACTRDeclarativeMemory(self.config)
        self.procedural_memory = ACTRProceduralMemory(self.config)
        self.utility_calculator = UtilityCalculator(self.config)
        self.tabu_search = TabuSearch(self.config)
        self.noise_generator = NoiseGenerator(self.config.noise_sigma)
        
        # State
        self.current_goal: Optional[Dict[str, Any]] = None
        self.cycle_count = 0
        
        # Callbacks for LLM estimation
        self.probability_estimator: Optional[Callable] = None
        self.cost_estimator: Optional[Callable] = None
    
    def set_llm_estimators(
        self,
        probability_estimator: Callable,
        cost_estimator: Callable
    ):
        """Set LLM-based estimators."""
        self.probability_estimator = probability_estimator
        self.cost_estimator = cost_estimator
    
    def add_chunk(self, chunk: ACTRChunk) -> str:
        """Add a chunk to declarative memory."""
        return self.declarative_memory.add_chunk(chunk)
    
    def add_production(self, production: ACTRProduction) -> str:
        """Add a production rule."""
        return self.procedural_memory.add_production(production)
    
    def run_cycle(self, context: Dict[str, Any]) -> Optional[ACTRProduction]:
        """
        Run one ACT-R cycle:
        1. Match productions to context
        2. Calculate utilities
        3. Select best operator
        4. Update history
        
        Returns:
            Selected operator or None
        """
        self.cycle_count += 1
        
        # Find matching productions
        matches = self.procedural_memory.find_matching_productions(context)
        
        if not matches:
            logger.debug("No matching productions")
            return None
        
        # Update probabilities and costs using LLM if available
        for op in matches:
            if self.probability_estimator:
                op.probability = self.utility_calculator.estimate_probability(
                    op, self.current_goal or {}, self.probability_estimator
                )
            if self.cost_estimator:
                op.cost = self.utility_calculator.estimate_cost(
                    op, self.cost_estimator
                )
        
        # Select operator
        selected = self.utility_calculator.select_operator(
            matches,
            self.current_goal or {},
            context
        )
        
        return selected
    
    def update_history(self, operator_id: str):
        """Add operator to tabu list."""
        self.utility_calculator.update_history(operator_id)
        self.tabu_search.add_to_history(operator_id)
    
    def retrieve_from_memory(
        self,
        pattern: Dict[str, Any],
        chunk_type: Optional[str] = None
    ) -> Optional[ACTRChunk]:
        """Retrieve chunk from declarative memory."""
        return self.declarative_memory.retrieve_by_pattern(pattern, chunk_type)
    
    def set_goal(self, goal: Dict[str, Any]):
        """Set the current goal."""
        self.current_goal = goal
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "cycle_count": self.cycle_count,
            "declarative_chunks": len(self.declarative_memory.chunks),
            "procedural_rules": len(self.procedural_memory.productions),
            "tabu_list_size": len(self.tabu_search.tabu_list.entries)
        }

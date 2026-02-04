"""Knowledge Engine DTS Integration.

Multi-turn conversation optimization for Knowledge Graph interactions.
Uses DTS to optimize dialog-based knowledge extraction and explanation.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Import DTS components
from integrations.dts import (
    DTSEngine,
    DTSConfig,
    DTSResult,
    ConversationTree,
    ConversationNode,
    ScoreResult,
    UserPersona,
    PREDEFINED_PERSONAS,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulatedResponse:
    """A simulated user response in KG interaction.
    
    Attributes:
        response: The response text
        intent: User intent type
        confidence: Confidence in the response
        persona: Persona that generated this response
        metadata: Additional response data
    """
    response: str
    intent: str
    confidence: float
    persona: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response": self.response,
            "intent": self.intent,
            "confidence": self.confidence,
            "persona": self.persona,
            **self.metadata,
        }


@dataclass
class ExtractedEntities:
    """Entities extracted via dialog.
    
    Attributes:
        entities: Dict of entity types to lists
        confidence: Extraction confidence
        extraction_path: Conversation path that led to extraction
        method: Extraction method used
    """
    entities: Dict[str, List[str]] = field(default_factory=dict)
    confidence: float = 0.0
    extraction_path: List[Dict[str, str]] = field(default_factory=list)
    method: str = "dts_dialog"
    
    def get_all_entities(self) -> List[str]:
        """Get all extracted entities as flat list."""
        all_entities = []
        for entity_list in self.entities.values():
            all_entities.extend(entity_list)
        return all_entities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities": self.entities,
            "confidence": self.confidence,
            "extraction_path": self.extraction_path,
            "method": self.method,
            "total_count": len(self.get_all_entities()),
        }


@dataclass
class ConversationScript:
    """A generated conversation script.
    
    Attributes:
        turns: List of conversation turns
        goal: Conversation goal
        estimated_effectiveness: Predicted effectiveness score
        personas_targeted: User personas this script targets
    """
    turns: List[Dict[str, str]] = field(default_factory=list)
    goal: str = ""
    estimated_effectiveness: float = 0.0
    personas_targeted: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turns": self.turns,
            "goal": self.goal,
            "estimated_effectiveness": self.estimated_effectiveness,
            "personas_targeted": self.personas_targeted,
            "turn_count": len(self.turns),
        }


@dataclass
class OptimalPath:
    """Optimal path for multi-turn retrieval.
    
    Attributes:
        steps: List of retrieval steps
        expected_outcome: Description of expected result
        confidence: Confidence in optimality
        alternative_paths: Backup paths
    """
    steps: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    confidence: float = 0.0
    alternative_paths: List[List[str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "steps": self.steps,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "alternative_paths": self.alternative_paths,
        }


class DTSKGIntegration:
    """Knowledge Engine DTS Integration.
    
    Multi-turn conversation optimization for Knowledge Graph interactions.
    Uses DTS to:
    - Optimize knowledge extraction dialogs
    - Simulate user interactions with KG queries
    - Score and improve KG conversation trajectories
    - Generate explanation trees for complex KG results
    - Backtrack and replan failed interactions
    
    Example:
        >>> integration = DTSKGIntegration()
        >>> 
        >>> # Optimize a knowledge extraction dialog
        >>> result = integration.optimize_kg_query_dialog(
        ...     context="Find all related companies",
        ...     user_goal="Research corporate connections"
        ... )
        >>> 
        >>> # Extract entities through optimized dialog
        >>> entities = integration.extract_kg_via_dialog(
        ...     entity_query="technology companies in AI"
        ... )
    """
    
    def __init__(
        self,
        dts_engine: Optional[DTSEngine] = None,
        config: Optional[DTSConfig] = None,
        kg_client: Optional[Any] = None,
    ):
        """Initialize KG-DTS integration.
        
        Args:
            dts_engine: Pre-configured DTS engine
            config: DTS configuration (if engine not provided)
            kg_client: Knowledge graph client
        """
        self.engine = dts_engine or DTSEngine(config=config or DTSConfig())
        self.kg_client = kg_client
        self._session_history: List[Dict[str, Any]] = []
        
        logger.info("DTS-KG Integration initialized")
    
    def optimize_kg_query_dialog(
        self,
        context: str,
        user_goal: str,
        kg_schema_hints: Optional[Dict[str, Any]] = None,
        rounds: int = 3,
    ) -> ConversationTree:
        """Optimize a dialog for KG query construction.
        
        Uses DTS to find the best conversation flow for helping
        a user build and refine a KG query.
        
        Args:
            context: Initial conversation context
            user_goal: What the user wants to achieve
            kg_schema_hints: Optional KG schema information
            rounds: Optimization rounds
            
        Returns:
            Optimized conversation tree
        """
        # Enhance context with KG-specific guidance
        enhanced_context = self._build_kg_context(context, kg_schema_hints)
        
        # Run DTS optimization
        result = self.engine.optimize_conversation(
            initial_context=enhanced_context,
            goal=user_goal,
            rounds=rounds,
        )
        
        # Log session
        self._log_session("optimize_kg_query_dialog", {
            "context": context,
            "user_goal": user_goal,
            "best_score": result.best_score,
        })
        
        return result.tree
    
    def simulate_user_interactions(
        self,
        query_plan: Dict[str, Any],
        num_variants: int = 3,
        persona_names: Optional[List[str]] = None,
    ) -> List[SimulatedResponse]:
        """Simulate user interactions with a KG query plan.
        
        Args:
            query_plan: The planned KG query
            num_variants: Number of user variants to simulate
            persona_names: Specific personas to use
            
        Returns:
            List of simulated responses
        """
        query_description = query_plan.get("description", str(query_plan))
        
        # Select personas
        if persona_names:
            personas = [
                p for name, p in PREDEFINED_PERSONAS.items()
                if name in persona_names
            ]
        else:
            personas = list(PREDEFINED_PERSONAS.values())[:num_variants]
        
        simulated = []
        for i, persona in enumerate(personas[:num_variants]):
            response = self.engine.user_sim.simulate_response(
                strategy=query_description,
                persona=persona,
            )
            
            # Classify intent
            intent, confidence = self.engine.user_sim.intent_model.classify_intent(response)
            
            simulated.append(SimulatedResponse(
                response=response,
                intent=intent.value,
                confidence=confidence,
                persona=persona.name,
                metadata={
                    "query_plan": query_plan,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ))
        
        return simulated
    
    def score_kg_trajectory(
        self,
        trajectory: List[ConversationNode],
    ) -> ScoreResult:
        """Score a KG interaction trajectory.
        
        Args:
            trajectory: Conversation path to score
            
        Returns:
            Score result with KG-specific criteria
        """
        # Use standard scorer
        result = self.engine.scorer.score_trajectory(trajectory)
        
        # Enhance with KG-specific evaluation
        kg_criteria = self._evaluate_kg_specific(trajectory)
        result.criteria_scores.update(kg_criteria)
        
        return result
    
    def generate_explanation_tree(
        self,
        kg_data: Dict[str, Any],
        target_audience: str = "general",
        depth: int = 3,
    ) -> ConversationTree:
        """Generate an explanation tree for KG query results.
        
        Creates an optimal conversation flow for explaining
        complex knowledge graph results to users.
        
        Args:
            kg_data: KG query results to explain
            target_audience: Target audience type
            depth: Maximum explanation depth
            
        Returns:
            Conversation tree for explanation
        """
        # Build explanation goal
        explanation_goal = self._build_explanation_goal(kg_data, target_audience)
        
        # Create initial context
        context = f"Explain KG results: {len(kg_data)} entities/relations found"
        
        # Run DTS with explanation-specific config
        result = self.engine.optimize_conversation(
            initial_context=context,
            goal=explanation_goal,
            rounds=depth,
        )
        
        # Mark tree as explanation type
        result.tree.metadata["type"] = "kg_explanation"
        result.tree.metadata["kg_data_summary"] = str(kg_data)[:200]
        
        return result.tree
    
    def backtrack_and_replan(
        self,
        failed_path: List[ConversationNode],
        failure_reason: str = "",
    ) -> ConversationTree:
        """Backtrack from a failed path and replan.
        
        Analyzes why a conversation failed and generates
        an improved strategy.
        
        Args:
            failed_path: The conversation path that failed
            failure_reason: Description of why it failed
            
        Returns:
        New conversation tree with improved strategy
        """
        # Analyze failure
        failed_turns = [
            {"speaker": n.speaker, "message": n.message}
            for n in failed_path
        ]
        
        # Create replanning context
        context = f"Previous attempt failed: {failure_reason}. "
        context += f"Failed after {len(failed_path)} turns."
        
        # Extract original goal
        goal = "Improve strategy based on failure"
        if failed_path and failed_path[0].metadata:
            goal = failed_path[0].metadata.get("goal", goal)
        
        # Run DTS with failure-aware config
        config = DTSConfig(
            beam_width=7,  # Wider search to find alternatives
            prune_threshold=4.0,  # Lower threshold to explore more
            max_depth=5,
        )
        
        temp_engine = DTSEngine(config=config)
        result = temp_engine.optimize_conversation(
            initial_context=context,
            goal=goal,
            rounds=3,
        )
        
        # Mark as replan
        result.tree.metadata["replan"] = True
        result.tree.metadata["failure_reason"] = failure_reason
        result.tree.metadata["original_path_length"] = len(failed_path)
        
        return result.tree
    
    def extract_kg_via_dialog(
        self,
        entity_query: str,
        entity_types: Optional[List[str]] = None,
    ) -> ExtractedEntities:
        """Extract KG entities through optimized dialog.
        
        Uses DTS-optimized conversation to guide entity extraction.
        
        Args:
            entity_query: Natural language entity query
            entity_types: Types of entities to extract
            
        Returns:
            Extracted entities with confidence
        """
        # Optimize extraction dialog
        result = self.engine.optimize_conversation(
            initial_context=f"Extract entities: {entity_query}",
            goal=f"Identify all relevant entities of types: {entity_types or 'any'}",
            rounds=3,
        )
        
        # Extract entities from best path (simulated)
        # In real implementation, this would use NLP on the path
        entities = self._extract_from_path(result.best_path, entity_types)
        
        return ExtractedEntities(
            entities=entities,
            confidence=result.best_score / 10.0,
            extraction_path=result.get_conversation_script(),
            method="dts_dialog_extraction",
        )
    
    def explain_kg_result_conversation(
        self,
        kg_data: Dict[str, Any],
        user_knowledge_level: str = "intermediate",
    ) -> ConversationScript:
        """Generate a conversation script for explaining KG results.
        
        Args:
            kg_data: KG data to explain
            user_knowledge_level: User's knowledge level
            
        Returns:
            Generated conversation script
        """
        # Generate explanation tree
        tree = self.generate_explanation_tree(
            kg_data=kg_data,
            target_audience=user_knowledge_level,
        )
        
        # Get best path as script
        best_path = tree.get_best_path()
        
        # Score the path
        score_result = self.score_kg_trajectory(best_path)
        
        return ConversationScript(
            turns=[
                {"speaker": n.speaker, "message": n.message}
                for n in best_path
            ],
            goal="Explain KG results",
            estimated_effectiveness=score_result.overall_score / 10.0,
            personas_targeted=[user_knowledge_level],
        )
    
    def optimize_multi_turn_retrieval(
        self,
        retrieval_goal: str,
        available_operations: Optional[List[str]] = None,
    ) -> OptimalPath:
        """Optimize multi-turn retrieval strategy.
        
        Finds the optimal sequence of KG operations to achieve
        a retrieval goal.
        
        Args:
            retrieval_goal: What to retrieve
            available_operations: Available KG operations
            
        Returns:
            Optimal retrieval path
        """
        context = f"Multi-turn retrieval: {retrieval_goal}"
        if available_operations:
            context += f". Available: {', '.join(available_operations)}"
        
        # Run optimization
        result = self.engine.optimize_conversation(
            initial_context=context,
            goal=retrieval_goal,
            rounds=4,
        )
        
        # Extract steps from best path
        steps = []
        for node in result.best_path:
            if node.speaker == "system":
                steps.append(node.message)
        
        # Get alternative paths (top 3 branches)
        alternatives = []
        for branch in result.tree.get_branches()[:3]:
            alt_steps = [
                n.message for n in branch 
                if n.speaker == "system"
            ]
            if alt_steps != steps:
                alternatives.append(alt_steps)
        
        return OptimalPath(
            steps=steps,
            expected_outcome=f"Retrieve: {retrieval_goal}",
            confidence=result.best_score / 10.0,
            alternative_paths=alternatives,
        )
    
    def _build_kg_context(
        self,
        base_context: str,
        schema_hints: Optional[Dict[str, Any]],
    ) -> str:
        """Build KG-enhanced context."""
        context = base_context
        
        if schema_hints:
            context += "\nKG Schema:"
            for entity_type, properties in schema_hints.items():
                context += f"\n  - {entity_type}: {properties}"
        
        return context
    
    def _build_explanation_goal(
        self,
        kg_data: Dict[str, Any],
        audience: str,
    ) -> str:
        """Build explanation goal from KG data."""
        entity_count = len(kg_data.get("entities", []))
        relation_count = len(kg_data.get("relations", []))
        
        goal = f"Explain {entity_count} entities and {relation_count} relations"
        goal += f" to {audience} audience"
        
        return goal
    
    def _evaluate_kg_specific(
        self,
        trajectory: List[ConversationNode],
    ) -> Dict[str, float]:
        """Evaluate KG-specific criteria."""
        scores = {}
        
        # Query specificity
        specificity = 5.0
        for node in trajectory:
            msg = node.message.lower()
            if any(word in msg for word in ["specific", "exact", "precise"]):
                specificity += 0.5
        scores["query_specificity"] = min(10.0, specificity)
        
        # Schema alignment
        alignment = 5.0
        for node in trajectory:
            if "entity" in node.message.lower() or "relation" in node.message.lower():
                alignment += 0.5
        scores["schema_alignment"] = min(10.0, alignment)
        
        return scores
    
    def _extract_from_path(
        self,
        path: List[ConversationNode],
        entity_types: Optional[List[str]],
    ) -> Dict[str, List[str]]:
        """Extract entities from conversation path."""
        # Simulated extraction - real implementation would use NLP
        entities = {}
        
        if entity_types:
            for etype in entity_types:
                entities[etype] = [f"{etype}_example_{i}" for i in range(2)]
        else:
            entities["unknown"] = ["extracted_entity_1", "extracted_entity_2"]
        
        return entities
    
    def _log_session(self, operation: str, details: Dict[str, Any]) -> None:
        """Log session operation."""
        self._session_history.append({
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        })
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get session history."""
        return self._session_history.copy()
    
    def reset_session(self) -> None:
        """Reset session history."""
        self._session_history = []


# Type hint for Any
from typing import Any

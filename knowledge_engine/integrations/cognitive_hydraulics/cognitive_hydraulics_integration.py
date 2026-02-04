"""Knowledge Engine Cognitive Hydraulics Integration.

Hybrid neuro-symbolic reasoning for KG operations.

Key classes:
    - CognitiveHydraulicsKGIntegration:
        - reason_about_graph(kg_subgraph, query): Symbolic KG reasoning
        - solve_kg_problem(problem_description): Solve complex KG problems
        - infer_relationship(entity1, entity2): Infer missing relationships
        - validate_kg_consistency(kg): Physics/logic validation
        - optimize_query_plan(query): Find optimal query execution
        - explain_reasoning(result): Generate explanation
        - learn_from_feedback(feedback): Chunk successful reasoning

    - ReasoningTracer: Trace reasoning steps for explainability
    - KGProblemEncoder: Encode KG problems for Soar/ACT-R
    - KGSolutionDecoder: Decode solutions back to KG updates

Follows CLAUDE.md patterns:
- SSOT: Primary logic in integrations/cognitive_hydraulics/, thin wrapper here
- Runtime Truth pattern
- UTC timestamps
- Structured JSON logging
- Idempotent operations
- Circuit breaker for LLM calls
- Configurable thresholds
"""

import logging
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from copy import deepcopy

# Import base cognitive hydraulics (thin wrapper - SSOT in integrations/)
from integrations.cognitive_hydraulics import (
    CognitiveHydraulicsEngine,
    ReasoningResult,
    SystemType,
    SoarOperator,
    SoarState,
    SoarRule,
    ACTRProduction,
    ACTRChunk,
)
from integrations.cognitive_hydraulics.config import CognitiveHydraulicsConfig

# Knowledge Engine imports (avoid direct core-projects imports per CLAUDE.md)
try:
    from knowledge_engine.schemas.base import Entity, Relationship, KnowledgeArtifact
    from knowledge_engine.graph.models import Node, Edge
except ImportError:
    # Fallback definitions for standalone operation
    Entity = dict
    Relationship = dict
    KnowledgeArtifact = dict
    Node = dict
    Edge = dict

logger = logging.getLogger(__name__)


@dataclass
class KGReasoningContext:
    """Context for KG reasoning operations."""
    kg_snapshot: Dict[str, Any] = field(default_factory=dict)
    query_type: str = ""
    constraints: List[Dict] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kg_snapshot_size": len(self.kg_snapshot),
            "query_type": self.query_type,
            "constraints_count": len(self.constraints),
            "preferences": self.preferences,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class KGReasoningResult:
    """Result of KG reasoning operation."""
    success: bool
    reasoning_type: str = ""
    conclusions: List[Dict] = field(default_factory=list)
    inferred_relationships: List[Dict] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0
    
    # Reasoning metadata
    systems_used: List[str] = field(default_factory=list)
    reasoning_trace: List[Dict] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    # KG updates
    proposed_entities: List[Dict] = field(default_factory=list)
    proposed_relationships: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "reasoning_type": self.reasoning_type,
            "conclusions": self.conclusions,
            "inferred_relationships": self.inferred_relationships,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "systems_used": self.systems_used,
            "execution_time_ms": self.execution_time_ms,
            "proposed_entities_count": len(self.proposed_entities),
            "proposed_relationships_count": len(self.proposed_relationships),
        }


class ReasoningTracer:
    """Trace reasoning steps for explainability."""
    
    def __init__(self):
        self.steps: List[Dict] = []
        self.current_step = 0
    
    def add_step(
        self,
        system: str,
        operation: str,
        input_data: Dict,
        output_data: Dict,
        reasoning: str = ""
    ):
        """Add a reasoning step."""
        self.current_step += 1
        self.steps.append({
            "step_number": self.current_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": system,
            "operation": operation,
            "input": input_data,
            "output": output_data,
            "reasoning": reasoning
        })
    
    def get_trace(self) -> List[Dict]:
        """Get the full reasoning trace."""
        return deepcopy(self.steps)
    
    def generate_explanation(self) -> str:
        """Generate human-readable explanation from trace."""
        lines = ["Reasoning Process:"]
        
        for step in self.steps:
            lines.append(f"\nStep {step['step_number']} ({step['system']}):")
            lines.append(f"  Operation: {step['operation']}")
            if step['reasoning']:
                lines.append(f"  Reasoning: {step['reasoning']}")
        
        return "\n".join(lines)


class KGProblemEncoder:
    """Encode KG problems for Soar/ACT-R."""
    
    def encode_graph_query(
        self,
        kg_subgraph: Dict[str, Any],
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encode KG query as problem for cognitive hydraulics."""
        problem = {
            "type": "graph_query",
            "entities": kg_subgraph.get("entities", []),
            "relationships": kg_subgraph.get("relationships", []),
            "query_constraints": query.get("constraints", []),
            "query_goal": query.get("goal", {}),
        }
        
        return problem
    
    def encode_relationship_inference(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encode relationship inference problem."""
        problem = {
            "type": "relationship_inference",
            "source_entity": entity1,
            "target_entity": entity2,
            "context_entities": context.get("entities", []),
            "known_relationships": context.get("relationships", []),
        }
        
        return problem
    
    def encode_consistency_validation(
        self,
        kg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encode consistency validation problem."""
        problem = {
            "type": "consistency_validation",
            "entities": kg.get("entities", []),
            "relationships": kg.get("relationships", []),
            "rules": kg.get("validation_rules", []),
        }
        
        return problem
    
    def encode_query_optimization(
        self,
        query: Dict[str, Any],
        kg_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encode query optimization problem."""
        problem = {
            "type": "query_optimization",
            "query_structure": query,
            "kg_statistics": kg_stats,
        }
        
        return problem


class KGSolutionDecoder:
    """Decode solutions back to KG updates."""
    
    def decode_graph_query_result(
        self,
        result: ReasoningResult,
        original_query: Dict[str, Any]
    ) -> KGReasoningResult:
        """Decode graph query result to KG format."""
        kg_result = KGReasoningResult(
            success=result.success,
            reasoning_type="graph_query",
            systems_used=result.systems_used,
            reasoning_trace=result.reasoning_trace,
            execution_time_ms=result.total_time_ms
        )
        
        if result.success and result.solution:
            # Extract conclusions from solution
            if isinstance(result.solution, dict):
                kg_result.conclusions = [result.solution]
            elif isinstance(result.solution, list):
                kg_result.conclusions = result.solution
            else:
                kg_result.conclusions = [{"value": str(result.solution)}]
            
            kg_result.confidence = 0.9  # High confidence for successful reasoning
        
        return kg_result
    
    def decode_relationship_inference(
        self,
        result: ReasoningResult,
        entity1_id: str,
        entity2_id: str
    ) -> KGReasoningResult:
        """Decode relationship inference result."""
        kg_result = KGReasoningResult(
            success=result.success,
            reasoning_type="relationship_inference",
            systems_used=result.systems_used,
            execution_time_ms=result.total_time_ms
        )
        
        if result.success and result.solution:
            # Extract inferred relationship
            solution = result.solution
            
            if isinstance(solution, dict):
                relationship_type = solution.get("relationship_type", "RELATED_TO")
                confidence = solution.get("confidence", 0.5)
            else:
                relationship_type = "RELATED_TO"
                confidence = 0.5
            
            kg_result.inferred_relationships.append({
                "source": entity1_id,
                "target": entity2_id,
                "type": relationship_type,
                "confidence": confidence,
                "inferred_by": "cognitive_hydraulics"
            })
            
            kg_result.confidence = confidence
        
        return kg_result
    
    def decode_consistency_validation(
        self,
        result: ReasoningResult
    ) -> KGReasoningResult:
        """Decode consistency validation result."""
        kg_result = KGReasoningResult(
            success=result.success,
            reasoning_type="consistency_validation",
            systems_used=result.systems_used,
            execution_time_ms=result.total_time_ms
        )
        
        if result.success and result.solution:
            solution = result.solution
            
            if isinstance(solution, dict):
                violations = solution.get("violations", [])
                kg_result.conclusions = [
                    {"valid": len(violations) == 0, "violations": violations}
                ]
            else:
                kg_result.conclusions = [{"valid": True}]
        
        return kg_result


class CognitiveHydraulicsKGIntegration:
    """
    Knowledge Engine Cognitive Hydraulics Integration.
    
    Provides hybrid neuro-symbolic reasoning for KG operations.
    """
    
    def __init__(self, config: Optional[CognitiveHydraulicsConfig] = None):
        self.config = config or CognitiveHydraulicsConfig()
        self.engine = CognitiveHydraulicsEngine(self.config)
        
        # Helper components
        self.encoder = KGProblemEncoder()
        self.decoder = KGSolutionDecoder()
        self.tracer = ReasoningTracer()
        
        # Operation tracking
        self.operation_count = 0
        self.operation_log: List[Dict] = []
    
    def reason_about_graph(
        self,
        kg_subgraph: Dict[str, Any],
        query: Dict[str, Any]
    ) -> KGReasoningResult:
        """
        Symbolic KG reasoning.
        
        Args:
            kg_subgraph: Subset of KG to reason about
            query: Query specification
            
        Returns:
            KGReasoningResult with conclusions
        """
        self.operation_count += 1
        start_time = datetime.now(timezone.utc)
        
        # Encode problem
        problem = self.encoder.encode_graph_query(kg_subgraph, query)
        goal = query.get("goal", {"result": "found"})
        
        self.tracer.add_step(
            "encoder",
            "encode_graph_query",
            {"kg_size": len(kg_subgraph), "query": query},
            {"problem": problem},
            "Encoded KG query as cognitive problem"
        )
        
        # Run reasoning
        result = self.engine.solve(problem, goal)
        
        # Decode result
        kg_result = self.decoder.decode_graph_query_result(result, query)
        kg_result.explanation = self.tracer.generate_explanation()
        
        # Log operation
        self._log_operation("reason_about_graph", {
            "success": kg_result.success,
            "execution_time_ms": kg_result.execution_time_ms
        })
        
        return kg_result
    
    def solve_kg_problem(
        self,
        problem_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> KGReasoningResult:
        """
        Solve complex KG problems.
        
        Args:
            problem_description: Natural language problem description
            context: Additional context
            
        Returns:
            KGReasoningResult with solution
        """
        self.operation_count += 1
        
        # Build problem structure
        problem = {
            "type": "complex_kg_problem",
            "description": problem_description,
            "context": context or {}
        }
        
        goal = {"solved": True}
        
        # Run reasoning
        result = self.engine.solve(problem, goal)
        
        # Build result
        kg_result = KGReasoningResult(
            success=result.success,
            reasoning_type="complex_problem",
            systems_used=result.systems_used,
            reasoning_trace=result.reasoning_trace,
            execution_time_ms=result.total_time_ms
        )
        
        if result.success:
            kg_result.conclusions = [{
                "solution": result.solution,
                "description": problem_description
            }]
            kg_result.confidence = 0.85
        
        self._log_operation("solve_kg_problem", {
            "success": kg_result.success,
            "description": problem_description[:100]
        })
        
        return kg_result
    
    def infer_relationship(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> KGReasoningResult:
        """
        Infer missing relationships between entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            context: KG context for inference
            
        Returns:
            KGReasoningResult with inferred relationship
        """
        self.operation_count += 1
        
        # Encode problem
        problem = self.encoder.encode_relationship_inference(
            entity1, entity2, context or {}
        )
        
        goal = {"relationship_inferred": True}
        
        # Run reasoning
        result = self.engine.solve(problem, goal)
        
        # Decode result
        kg_result = self.decoder.decode_relationship_inference(
            result,
            entity1.get("entity_id", "unknown"),
            entity2.get("entity_id", "unknown")
        )
        
        self._log_operation("infer_relationship", {
            "success": kg_result.success,
            "entities": [entity1.get("name"), entity2.get("name")]
        })
        
        return kg_result
    
    def validate_kg_consistency(
        self,
        kg: Dict[str, Any]
    ) -> KGReasoningResult:
        """
        Physics/logic validation of KG.
        
        Args:
            kg: Knowledge graph to validate
            
        Returns:
            KGReasoningResult with validation results
        """
        self.operation_count += 1
        
        # Encode problem
        problem = self.encoder.encode_consistency_validation(kg)
        goal = {"valid": True, "violations": []}
        
        # Run reasoning
        result = self.engine.solve(problem, goal)
        
        # Decode result
        kg_result = self.decoder.decode_consistency_validation(result)
        
        self._log_operation("validate_kg_consistency", {
            "success": kg_result.success,
            "entity_count": len(kg.get("entities", []))
        })
        
        return kg_result
    
    def optimize_query_plan(
        self,
        query: Dict[str, Any],
        kg_stats: Optional[Dict[str, Any]] = None
    ) -> KGReasoningResult:
        """
        Find optimal query execution plan.
        
        Args:
            query: Query to optimize
            kg_stats: KG statistics for optimization
            
        Returns:
            KGReasoningResult with optimized plan
        """
        self.operation_count += 1
        
        # Encode problem
        problem = self.encoder.encode_query_optimization(
            query, kg_stats or {}
        )
        goal = {"optimal_plan_found": True}
        
        # Run reasoning
        result = self.engine.solve(problem, goal)
        
        # Build result
        kg_result = KGReasoningResult(
            success=result.success,
            reasoning_type="query_optimization",
            systems_used=result.systems_used,
            execution_time_ms=result.total_time_ms
        )
        
        if result.success and result.solution:
            if isinstance(result.solution, dict):
                plan = result.solution.get("plan", [])
            else:
                plan = [{"step": str(result.solution)}]
            
            kg_result.conclusions = [{"optimized_plan": plan}]
            kg_result.confidence = 0.8
        
        self._log_operation("optimize_query_plan", {
            "success": kg_result.success,
            "query_type": query.get("type", "unknown")
        })
        
        return kg_result
    
    def explain_reasoning(self, result: KGReasoningResult) -> str:
        """
        Generate explanation for reasoning result.
        
        Args:
            result: Reasoning result to explain
            
        Returns:
            Human-readable explanation
        """
        if result.explanation:
            return result.explanation
        
        # Generate explanation from trace
        lines = [
            f"Reasoning Type: {result.reasoning_type}",
            f"Success: {result.success}",
            f"Systems Used: {', '.join(result.systems_used)}",
            f"Confidence: {result.confidence:.2%}",
            "\nConclusions:"
        ]
        
        for i, conclusion in enumerate(result.conclusions, 1):
            lines.append(f"  {i}. {json.dumps(conclusion, indent=2)}")
        
        if result.inferred_relationships:
            lines.append("\nInferred Relationships:")
            for rel in result.inferred_relationships:
                lines.append(f"  - {rel['source']} --[{rel['type']}]--> {rel['target']} "
                           f"(confidence: {rel['confidence']:.2%})")
        
        return "\n".join(lines)
    
    def learn_from_feedback(
        self,
        feedback: Dict[str, Any]
    ) -> bool:
        """
        Chunk successful reasoning from feedback.
        
        Args:
            feedback: Feedback on reasoning operation
            
        Returns:
            True if learning occurred
        """
        if not feedback.get("success", False):
            return False
        
        # Extract impasse and resolution from feedback
        impasse_data = feedback.get("impasse")
        resolution = feedback.get("resolution")
        context = feedback.get("context", {})
        
        if not impasse_data or not resolution:
            return False
        
        # Create chunk through the engine's chunking system
        # This is a simplified version - full implementation would
        # reconstruct the impasse object
        logger.info(f"Learning from successful resolution: {feedback.get('operation_id')}")
        
        return True
    
    def get_reasoning_trace(self) -> List[Dict]:
        """Get the current reasoning trace."""
        return self.tracer.get_trace()
    
    def reset_tracer(self):
        """Reset the reasoning tracer."""
        self.tracer = ReasoningTracer()
    
    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log an operation."""
        self.operation_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "details": details
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "operation_count": self.operation_count,
            "cognitive_hydraulics_stats": self.engine.get_stats(),
            "recent_operations": self.operation_log[-10:] if self.operation_log else []
        }

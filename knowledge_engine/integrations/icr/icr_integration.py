"""Knowledge Engine ICR Integration.

Iterative refinement for Knowledge Graph extraction, query optimization,
entity resolution, and schema inference using the ICR engine.

This module bridges the core ICR system with Knowledge Graph operations,
enabling self-improving KG construction and maintenance.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timezone

from integrations.icr import (
    ICREngine,
    Generator,
    Critic,
    Refiner,
    Judge,
    Criteria,
    RefinementResult,
    CritiqueResult,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class RefinedExtraction:
    """Result of refined KG extraction.
    
    Attributes:
        entities: Extracted and refined entities
        relations: Extracted and refined relations
        confidence: Overall extraction confidence
        iterations: Number of refinement iterations performed
        improvement: Score improvement from refinement
    """
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    confidence: float
    iterations: int
    improvement: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovedQuery:
    """Result of improved Cypher query.
    
    Attributes:
        query: The improved Cypher query
        original_query: The original query
        improvements: List of improvements made
        performance_estimate: Estimated performance improvement
        confidence: Confidence in query correctness
    """
    query: str
    original_query: str
    improvements: List[str]
    performance_estimate: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinedEntities:
    """Result of refined entity resolution.
    
    Attributes:
        entities: Resolved entity clusters
        duplicates_found: Number of duplicates identified
        merges_performed: Number of entity merges
        confidence: Resolution confidence
    """
    entities: List[Dict[str, Any]]
    duplicates_found: int
    merges_performed: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizedKG:
    """Result of optimized KG structure.
    
    Attributes:
        nodes: Optimized node structure
        edges: Optimized edge structure
        optimizations: List of optimizations applied
        metrics: Performance metrics
    """
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    optimizations: List[str]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinedSchema:
    """Result of refined schema inference.
    
    Attributes:
        schema: The inferred and refined schema
        entity_types: Defined entity types
        relation_types: Defined relation types
        confidence: Schema confidence score
        coverage: Data coverage percentage
    """
    schema: Dict[str, Any]
    entity_types: List[str]
    relation_types: List[str]
    confidence: float
    coverage: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ICRKGIntegration:
    """Knowledge Engine ICR Integration.
    
    Provides iterative refinement capabilities specifically tailored for
    Knowledge Graph operations including:
    - Entity extraction refinement
    - Relation extraction refinement  
    - Cypher query optimization
    - Entity resolution
    - KG structure optimization
    - Schema inference
    
    Example:
        >>> icr_kg = ICRKGIntegration()
        >>> 
        >>> # Refine entity extraction
        >>> result = icr_kg.refine_entity_extraction(
        ...     text="Apple Inc. was founded by Steve Jobs.",
        ...     initial_entities=[{"name": "Apple", "type": "Company"}]
        ... )
        >>> print(f"Refined {len(result.entities)} entities with confidence {result.confidence}")
        >>>
        >>> # Improve a Cypher query
        >>> query_result = icr_kg.improve_cypher_query(
        ...     "MATCH (n) RETURN n LIMIT 10"
        ... )
        >>> print(query_result.query)
    """
    
    def __init__(
        self,
        engine: Optional[ICREngine] = None,
        kg_backend: Optional[Any] = None,
        max_iterations: int = 5,
        quality_threshold: float = 0.85,
    ):
        """Initialize ICR KG Integration.
        
        Args:
            engine: ICREngine instance (creates default if None)
            kg_backend: Knowledge Graph backend connector
            max_iterations: Default max refinement iterations
            quality_threshold: Default quality threshold
        """
        self.engine = engine or self._create_default_engine(
            max_iterations, quality_threshold
        )
        self.kg_backend = kg_backend
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self._operation_count = 0
        
        # KG-specific criteria
        self.extraction_criteria = Criteria(
            accuracy=0.30,
            completeness=0.25,
            clarity=0.15,
            conciseness=0.05,
            correctness=0.20,
            consistency=0.05,
        )
        
        self.query_criteria = Criteria(
            accuracy=0.25,
            completeness=0.15,
            clarity=0.20,
            conciseness=0.15,
            correctness=0.20,
            consistency=0.05,
        )
        
        logger.info("Initialized ICRKGIntegration")
    
    def _create_default_engine(
        self,
        max_iterations: int,
        threshold: float,
    ) -> ICREngine:
        """Create default ICR engine for KG operations."""
        return ICREngine(
            max_iterations=max_iterations,
            quality_threshold=threshold,
            early_stopping=True,
            patience=2,
        )
    
    def refine_kg_extraction(
        self,
        text: str,
        initial_extraction: Optional[Dict[str, List[Dict]]] = None,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> RefinedExtraction:
        """Refine KG extraction from text.
        
        Args:
            text: Source text for extraction
            initial_extraction: Initial extraction results (optional)
            entity_types: Expected entity types
            relation_types: Expected relation types
            
        Returns:
            RefinedExtraction with improved entities and relations
        """
        self._operation_count += 1
        logger.info("Refining KG extraction", extra={
            "text_length": len(text),
            "operation": "refine_kg_extraction",
        })
        
        # Generate initial extraction if not provided
        if initial_extraction is None:
            initial_extraction = self._generate_initial_extraction(
                text, entity_types, relation_types
            )
        
        # Format extraction for refinement
        extraction_text = self._format_extraction(initial_extraction)
        
        # Refine using ICR engine
        prompt = f"""Refine this knowledge graph extraction from the text:

Source Text:
{text[:500]}...

Current Extraction:
{extraction_text}

Improve by:
1. Adding missing entities
2. Correcting entity types
3. Adding missing relations
4. Resolving ambiguities
5. Improving confidence scores
"""
        
        result = self.engine.refine(
            prompt=prompt,
            initial_output=extraction_text,
            context={
                "criteria": self.extraction_criteria,
                "operation": "kg_extraction",
            },
        )
        
        # Parse refined extraction
        refined_entities, refined_relations = self._parse_extraction(result.final_output)
        
        initial_confidence = self._calculate_extraction_confidence(initial_extraction)
        final_confidence = self._calculate_extraction_confidence({
            "entities": refined_entities,
            "relations": refined_relations,
        })
        
        return RefinedExtraction(
            entities=refined_entities,
            relations=refined_relations,
            confidence=final_confidence,
            iterations=result.iterations,
            improvement=result.total_improvement,
            metadata={
                "original_confidence": initial_confidence,
                "refinement_result": result.to_dict(),
            },
        )
    
    def refine_entity_extraction(
        self,
        text: str,
        initial_entities: List[Dict[str, Any]],
        max_iterations: Optional[int] = None,
    ) -> RefinedExtraction:
        """Refine entity extraction specifically.
        
        Args:
            text: Source text
            initial_entities: Initially extracted entities
            max_iterations: Override max iterations
            
        Returns:
            RefinedExtraction with improved entities
        """
        return self.refine_kg_extraction(
            text=text,
            initial_extraction={"entities": initial_entities, "relations": []},
        )
    
    def refine_relation_extraction(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        initial_relations: List[Dict[str, Any]],
    ) -> RefinedExtraction:
        """Refine relation extraction between entities.
        
        Args:
            text: Source text
            entities: Known entities
            initial_relations: Initially extracted relations
            
        Returns:
            RefinedExtraction with improved relations
        """
        return self.refine_kg_extraction(
            text=text,
            initial_extraction={"entities": entities, "relations": initial_relations},
        )
    
    def improve_cypher_query(
        self,
        query: str,
        optimization_goals: Optional[List[str]] = None,
        schema_context: Optional[Dict[str, Any]] = None,
    ) -> ImprovedQuery:
        """Improve Cypher query performance and correctness.
        
        Args:
            query: Original Cypher query
            optimization_goals: Goals (e.g., ["performance", "readability"])
            schema_context: Schema information for validation
            
        Returns:
            ImprovedQuery with optimized query
        """
        self._operation_count += 1
        logger.info("Improving Cypher query", extra={
            "query_length": len(query),
            "operation": "improve_cypher_query",
        })
        
        goals = optimization_goals or ["performance", "correctness"]
        
        prompt = f"""Optimize this Cypher query:

Original Query:
```cypher
{query}
```

Optimization Goals:
{', '.join(goals)}

Improve by:
1. Adding appropriate indexes hints
2. Optimizing MATCH patterns
3. Reducing cardinality
4. Improving readability
5. Ensuring correctness
"""
        
        result = self.engine.refine(
            prompt=prompt,
            initial_output=query,
            context={
                "criteria": self.query_criteria,
                "operation": "cypher_optimization",
            },
        )
        
        # Extract improvements from critique history
        improvements = []
        for critique in result.critique_history:
            for issue in critique.issues:
                if issue.type.value in ["correctness", "clarity"]:
                    improvements.append(f"{issue.type.value}: {issue.description}")
        
        return ImprovedQuery(
            query=result.final_output.strip(),
            original_query=query,
            improvements=list(set(improvements))[:5],
            performance_estimate=0.8 + result.total_improvement * 0.2,
            confidence=result.final_score,
            metadata={
                "optimization_goals": goals,
                "iterations": result.iterations,
            },
        )
    
    def refine_entity_resolution(
        self,
        entities: List[Dict[str, Any]],
        resolution_threshold: float = 0.85,
    ) -> RefinedEntities:
        """Refine entity resolution to identify duplicates.
        
        Args:
            entities: List of entities to resolve
            resolution_threshold: Similarity threshold for merging
            
        Returns:
            RefinedEntities with resolved clusters
        """
        self._operation_count += 1
        logger.info("Refining entity resolution", extra={
            "entity_count": len(entities),
            "operation": "refine_entity_resolution",
        })
        
        # Format entities for refinement
        entities_text = self._format_entities_for_resolution(entities)
        
        prompt = f"""Resolve duplicate entities in this list:

Entities:
{entities_text}

Resolution Criteria:
- Same name (case insensitive)
- Similar names with same type
- Same aliases
- Threshold: {resolution_threshold}

Output merged entity clusters with canonical representatives.
"""
        
        result = self.engine.refine(
            prompt=prompt,
            initial_output=entities_text,
        )
        
        resolved_entities = self._parse_resolved_entities(result.final_output)
        
        duplicates = len(entities) - len(resolved_entities)
        
        return RefinedEntities(
            entities=resolved_entities,
            duplicates_found=max(0, duplicates),
            merges_performed=max(0, duplicates),
            confidence=result.final_score,
            metadata={
                "original_count": len(entities),
                "final_count": len(resolved_entities),
                "threshold": resolution_threshold,
            },
        )
    
    def optimize_kg_structure(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        optimization_criteria: Optional[List[str]] = None,
    ) -> OptimizedKG:
        """Optimize Knowledge Graph structure.
        
        Args:
            nodes: Current nodes
            edges: Current edges
            optimization_criteria: Criteria for optimization
            
        Returns:
            OptimizedKG with improved structure
        """
        self._operation_count += 1
        
        criteria = optimization_criteria or ["normalization", "indexing", "compression"]
        
        kg_summary = f"""Nodes: {len(nodes)}
Edges: {len(edges)}
Node types: {len(set(n.get('type', 'Unknown') for n in nodes))}
Edge types: {len(set(e.get('type', 'Unknown') for e in edges))}
"""
        
        prompt = f"""Optimize this Knowledge Graph structure:

{kg_summary}

Optimization Criteria:
{', '.join(criteria)}

Suggest structural improvements:
1. Node property normalization
2. Index recommendations
3. Compression opportunities
4. Schema alignment
"""
        
        result = self.engine.refine(
            prompt=prompt,
            initial_output=kg_summary,
        )
        
        # Parse optimizations
        optimizations = []
        for critique in result.critique_history:
            for suggestion in critique.suggestions:
                optimizations.append(suggestion.fix)
        
        return OptimizedKG(
            nodes=nodes,
            edges=edges,
            optimizations=list(set(optimizations))[:10],
            metrics={
                "optimization_score": result.final_score,
                "iterations": result.iterations,
                "improvement": result.total_improvement,
            },
            metadata={
                "criteria": criteria,
            },
        )
    
    def iterative_schema_inference(
        self,
        data: List[Dict[str, Any]],
        initial_schema: Optional[Dict[str, Any]] = None,
        max_iterations: Optional[int] = None,
    ) -> RefinedSchema:
        """Iteratively refine schema inference from data.
        
        Args:
            data: Sample data for schema inference
            initial_schema: Initial schema guess (optional)
            max_iterations: Override max iterations
            
        Returns:
            RefinedSchema with optimized schema
        """
        self._operation_count += 1
        
        # Generate initial schema if not provided
        if initial_schema is None:
            initial_schema = self._infer_initial_schema(data)
        
        schema_text = self._format_schema(initial_schema)
        
        prompt = f"""Refine this knowledge graph schema based on {len(data)} samples:

Current Schema:
{schema_text}

Improve by:
1. Adding missing entity types
2. Adding missing relation types  
3. Adding property constraints
4. Defining cardinality rules
5. Improving type specificity
"""
        
        result = self.engine.refine(
            prompt=prompt,
            initial_output=schema_text,
            max_iterations=max_iterations or self.max_iterations,
        )
        
        refined_schema = self._parse_schema(result.final_output)
        
        return RefinedSchema(
            schema=refined_schema,
            entity_types=list(refined_schema.get("entities", {}).keys()),
            relation_types=list(refined_schema.get("relations", {}).keys()),
            confidence=result.final_score,
            coverage=self._calculate_schema_coverage(refined_schema, data),
            metadata={
                "sample_size": len(data),
                "iterations": result.iterations,
            },
        )
    
    def improve_kg_quality(
        self,
        kg: Dict[str, Any],
        quality_criteria: Optional[Dict[str, float]] = None,
    ) -> RefinedExtraction:
        """General KG quality improvement.
        
        Args:
            kg: Knowledge graph dict with entities and relations
            quality_criteria: Quality criteria weights
            
        Returns:
            RefinedExtraction with improved KG
        """
        criteria = quality_criteria or {
            "completeness": 0.3,
            "accuracy": 0.3,
            "consistency": 0.2,
            "richness": 0.2,
        }
        
        kg_text = self._format_kg(kg)
        
        prompt = f"""Improve the quality of this knowledge graph:

{kg_text}

Quality Criteria:
{chr(10).join(f"- {k}: {v}" for k, v in criteria.items())}

Focus on improving the lowest-scoring criteria.
"""
        
        result = self.engine.refine(
            prompt=prompt,
            initial_output=kg_text,
        )
        
        entities, relations = self._parse_extraction(result.final_output)
        
        return RefinedExtraction(
            entities=entities,
            relations=relations,
            confidence=result.final_score,
            iterations=result.iterations,
            improvement=result.total_improvement,
            metadata={"quality_criteria": criteria},
        )
    
    def converge_to_optimal(
        self,
        initial: Any,
        judge_fn: Callable[[Any], float],
        max_iter: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generic convergence to optimal state.
        
        Args:
            initial: Initial state
            judge_fn: Function that scores a state (0-1)
            max_iter: Maximum iterations
            
        Returns:
            Dict with optimal state and convergence info
        """
        max_iter = max_iter or self.max_iterations
        
        current = initial
        best = current
        best_score = judge_fn(current)
        history = [best_score]
        
        for i in range(max_iter):
            # Refine current state
            current_str = str(current)
            result = self.engine.refine(
                prompt="Optimize this state for maximum quality",
                initial_output=current_str,
                max_iterations=1,
            )
            
            # Parse back (simplified - assumes string representation)
            current = result.final_output
            score = judge_fn(current)
            history.append(score)
            
            if score > best_score:
                best = current
                best_score = score
            
            # Check convergence
            if len(history) >= 3 and max(history[-3:]) - min(history[-3:]) < 0.01:
                break
        
        return {
            "optimal": best,
            "score": best_score,
            "iterations": len(history) - 1,
            "history": history,
        }
    
    # Helper methods
    def _generate_initial_extraction(
        self,
        text: str,
        entity_types: Optional[List[str]],
        relation_types: Optional[List[str]],
    ) -> Dict[str, List[Dict]]:
        """Generate initial extraction from text."""
        # Simple heuristic-based extraction
        entities = []
        relations = []
        
        # Extract capitalized phrases as potential entities
        import re
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
        for i, name in enumerate(set(capitalized)):
            entities.append({
                "id": f"ent_{i}",
                "name": name,
                "type": "Unknown",
                "confidence": 0.5,
            })
        
        return {"entities": entities, "relations": relations}
    
    def _format_extraction(self, extraction: Dict[str, List[Dict]]) -> str:
        """Format extraction for refinement."""
        lines = ["Entities:"]
        for ent in extraction.get("entities", []):
            lines.append(f"  - {ent.get('name', 'Unknown')} ({ent.get('type', 'Unknown')})")
        
        lines.append("\nRelations:")
        for rel in extraction.get("relations", []):
            lines.append(f"  - {rel.get('source', '?')} --{rel.get('type', '?')}--> {rel.get('target', '?')}")
        
        return "\n".join(lines)
    
    def _parse_extraction(self, text: str) -> tuple:
        """Parse extraction from refined text."""
        # Simplified parsing - in production would use structured output
        entities = []
        relations = []
        
        import re
        entity_pattern = r'-\s*(\w[\w\s]*)\s*\((\w+)\)'
        for match in re.finditer(entity_pattern, text):
            entities.append({
                "name": match.group(1).strip(),
                "type": match.group(2),
            })
        
        return entities, relations
    
    def _calculate_extraction_confidence(self, extraction: Dict) -> float:
        """Calculate overall extraction confidence."""
        entities = extraction.get("entities", [])
        if not entities:
            return 0.0
        
        confidences = [e.get("confidence", 0.5) for e in entities]
        return sum(confidences) / len(confidences)
    
    def _format_entities_for_resolution(self, entities: List[Dict]) -> str:
        """Format entities for resolution."""
        lines = []
        for e in entities:
            lines.append(f"- {e.get('name', 'Unknown')} ({e.get('type', 'Unknown')})")
        return "\n".join(lines)
    
    def _parse_resolved_entities(self, text: str) -> List[Dict]:
        """Parse resolved entities from text."""
        entities = []
        import re
        for match in re.finditer(r'-\s*(\w[\w\s]*)\s*\((\w+)\)', text):
            entities.append({
                "name": match.group(1).strip(),
                "type": match.group(2),
            })
        return entities
    
    def _infer_initial_schema(self, data: List[Dict]) -> Dict:
        """Infer initial schema from data."""
        entity_types = set()
        relation_types = set()
        
        for item in data:
            entity_types.add(item.get("type", "Unknown"))
            for rel in item.get("relations", []):
                relation_types.add(rel.get("type", "Unknown"))
        
        return {
            "entities": {t: {"properties": []} for t in entity_types},
            "relations": {t: {"from": "", "to": ""} for t in relation_types},
        }
    
    def _format_schema(self, schema: Dict) -> str:
        """Format schema for refinement."""
        lines = ["Entity Types:"]
        for ent_type, props in schema.get("entities", {}).items():
            lines.append(f"  - {ent_type}")
        
        lines.append("\nRelation Types:")
        for rel_type, info in schema.get("relations", {}).items():
            lines.append(f"  - {rel_type}")
        
        return "\n".join(lines)
    
    def _parse_schema(self, text: str) -> Dict:
        """Parse schema from refined text."""
        # Simplified parsing
        return {
            "entities": {},
            "relations": {},
            "raw": text,
        }
    
    def _calculate_schema_coverage(self, schema: Dict, data: List[Dict]) -> float:
        """Calculate schema coverage of data."""
        if not data:
            return 0.0
        
        # Count how many data items match the schema
        matched = 0
        entity_types = set(schema.get("entities", {}).keys())
        
        for item in data:
            if item.get("type") in entity_types:
                matched += 1
        
        return matched / len(data)
    
    def _format_kg(self, kg: Dict) -> str:
        """Format KG for quality improvement."""
        entities = kg.get("entities", [])
        relations = kg.get("relations", [])
        return f"Entities: {len(entities)}\nRelations: {len(relations)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "operations_performed": self._operation_count,
            "max_iterations": self.max_iterations,
            "quality_threshold": self.quality_threshold,
            "engine_stats": self.engine.get_stats(),
        }


class ICRKGError(Exception):
    """Error during ICR KG operation."""
    pass

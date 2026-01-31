"""
Z3 Knowledge Extraction Integration for OpenEvolve Knowledge Engine

Integrates Z3 solver knowledge extraction with the existing knowledge engine infrastructure:
- Automatic extraction from Z3 solver results
- Storage in knowledge engine databases
- Pattern matching and strategy recommendations
- Knowledge graph construction

Author: OpenEvolve
Created: 2026-01-31
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import uuid
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

# Import knowledge engine components
try:
    from knowledge_engine.data.storage import (
        KnowledgeStorageEngine,
        KnowledgeArtifact,
        DatabaseManager
    )
    KE_AVAILABLE = True
except ImportError:
    KE_AVAILABLE = False
    logger.warning("Knowledge engine storage not available")

try:
    from knowledge_engine.knowledge_extractor import (
        KnowledgeExtractor,
        KnowledgeArtifact as KEKnowledgeArtifact
    )
    KE_EXTRACTOR_AVAILABLE = True
except ImportError:
    KE_EXTRACTOR_AVAILABLE = False
    logger.warning("Knowledge engine extractor not available")

# Import Z3 knowledge extraction
try:
    from z3_knowledge_extraction import (
        Z3KnowledgeExtractor,
        ProofPattern,
        ConstraintPattern,
        SolutionStrategy,
        MathematicalInsight,
        get_z3_knowledge_extractor
    )
    Z3_KE_AVAILABLE = True
except ImportError:
    Z3_KE_AVAILABLE = False
    logger.warning("Z3 knowledge extraction not available")

# Import Z3 integration
try:
    from z3prover_integration import Z3SolverResult, Z3TheoremResult
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


@dataclass
class Z3KnowledgeEntry:
    """Z3-specific knowledge entry for storage."""
    entry_id: str
    entry_type: str  # 'proof_pattern', 'constraint', 'strategy', 'insight'
    content: str
    metadata: Dict[str, Any]
    source_problem: str
    confidence: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "type": self.entry_type,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source_problem,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }


class Z3KnowledgeIntegration:
    """
    Integrates Z3 knowledge extraction with OpenEvolve knowledge engine.
    
    Provides:
    - Automatic extraction from solver results
    - Unified storage in knowledge engine
    - Pattern matching and recommendations
    - Knowledge graph integration
    """
    
    def __init__(self, storage_engine: Optional[Any] = None):
        """
        Initialize Z3 knowledge integration.
        
        Args:
            storage_engine: Optional existing storage engine
        """
        self.storage = storage_engine
        self.z3_extractor = get_z3_knowledge_extractor() if Z3_KE_AVAILABLE else None
        
        # Statistics
        self.extraction_count = 0
        self.storage_count = 0
        
        logger.info({
            "msg": "Z3 Knowledge Integration initialized",
            "storage_available": self.storage is not None,
            "extractor_available": self.z3_extractor is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize storage if needed."""
        if self.storage is None and KE_AVAILABLE:
            config = config or self._default_config()
            self.storage = KnowledgeStorageEngine(config)
            await self.storage.initialize()
            logger.info("Z3 knowledge storage initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for storage."""
        return {
            "database": {
                "type": "sqlite",
                "database": "./z3_knowledge.db"
            },
            "vector_store": {
                "type": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection_name": "z3_knowledge"
            },
            "cache": {
                "type": "redis",
                "host": "localhost",
                "port": 6379,
                "ttl_seconds": 3600
            }
        }
    
    # ========================================================================
    # Knowledge Extraction from Z3 Results
    # ========================================================================
    
    async def extract_from_solver_result(
        self,
        result: Any,
        problem_statement: str,
        problem_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Extract knowledge from a Z3 solver result.
        
        Args:
            result: Z3SolverResult or Z3TheoremResult
            problem_statement: Original problem
            problem_type: Classification of problem
            
        Returns:
            Dictionary with extracted knowledge
        """
        if not self.z3_extractor:
            return {"error": "Z3 knowledge extractor not available"}
        
        extracted = {
            "insights": [],
            "patterns": [],
            "strategies": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Extract insights from solution
            if hasattr(result, 'model') and result.model:
                insights = self.z3_extractor.extract_insights(
                    result, problem_statement
                )
                extracted["insights"] = [i.to_dict() for i in insights]
            
            # Extract constraint patterns
            if hasattr(result, 'constraints'):
                patterns = self.z3_extractor.analyze_constraints(
                    result.constraints,
                    getattr(result, 'solving_time', 0.0),
                    getattr(result, 'success', False)
                )
                extracted["patterns"] = [p.to_dict() for p in patterns]
            
            # Learn strategy if successful
            if getattr(result, 'success', False):
                strategy = self.z3_extractor.learn_strategy(
                    problem_features={
                        "type": problem_type,
                        "var_count": len(getattr(result, 'model', {}).assignments or {}),
                        "constraint_count": len(getattr(result, 'constraints', []))
                    },
                    tactics_used=getattr(result, 'tactics_used', []),
                    config_used=getattr(result, 'config', {}),
                    success=True,
                    solving_time=getattr(result, 'solving_time', 0.0)
                )
                extracted["strategies"] = [strategy.to_dict()]
            
            self.extraction_count += 1
            
            logger.info({
                "msg": "Knowledge extracted from Z3 result",
                "insights_count": len(extracted["insights"]),
                "patterns_count": len(extracted["patterns"]),
                "strategies_count": len(extracted["strategies"]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error({
                "msg": "Failed to extract knowledge",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return extracted
    
    async def store_extracted_knowledge(
        self,
        extracted: Dict[str, Any],
        problem_id: str
    ) -> List[str]:
        """
        Store extracted knowledge in the knowledge engine.
        
        Args:
            extracted: Knowledge extracted from solver
            problem_id: Identifier for the problem
            
        Returns:
            List of stored artifact IDs
        """
        if not self.storage:
            logger.warning("Storage not available, skipping storage")
            return []
        
        artifact_ids = []
        
        try:
            # Store insights
            for insight in extracted.get("insights", []):
                artifact_id = await self._store_insight(insight, problem_id)
                if artifact_id:
                    artifact_ids.append(artifact_id)
            
            # Store patterns
            for pattern in extracted.get("patterns", []):
                artifact_id = await self._store_pattern(pattern, problem_id)
                if artifact_id:
                    artifact_ids.append(artifact_id)
            
            # Store strategies
            for strategy in extracted.get("strategies", []):
                artifact_id = await self._store_strategy(strategy, problem_id)
                if artifact_id:
                    artifact_ids.append(artifact_id)
            
            self.storage_count += len(artifact_ids)
            
            logger.info({
                "msg": "Extracted knowledge stored",
                "problem_id": problem_id,
                "artifacts_stored": len(artifact_ids),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error({
                "msg": "Failed to store knowledge",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return artifact_ids
    
    async def _store_insight(
        self,
        insight: Dict[str, Any],
        problem_id: str
    ) -> Optional[str]:
        """Store a mathematical insight."""
        try:
            content = insight.get("statement", "")
            metadata = {
                "category": insight.get("category"),
                "confidence": insight.get("confidence"),
                "formal_representation": insight.get("formal_representation"),
                "derived_from": insight.get("derived_from", [])
            }
            
            artifact_id = await self.storage.store_knowledge_artifact(
                content=content,
                artifact_type="z3_insight",
                source=f"z3_solver:{problem_id}",
                metadata=metadata,
                confidence=float(insight.get("confidence", "0.5").rstrip("%")) / 100
            )
            
            return artifact_id
            
        except Exception as e:
            logger.error(f"Failed to store insight: {e}")
            return None
    
    async def _store_pattern(
        self,
        pattern: Dict[str, Any],
        problem_id: str
    ) -> Optional[str]:
        """Store a constraint pattern."""
        try:
            content = f"Pattern type: {pattern.get('type', 'unknown')}\n"
            content += f"Structure: {pattern.get('structure', '')}"
            
            metadata = {
                "pattern_type": pattern.get("type"),
                "variables": pattern.get("variables", []),
                "complexity": pattern.get("complexity"),
                "frequency": pattern.get("frequency")
            }
            
            artifact_id = await self.storage.store_knowledge_artifact(
                content=content,
                artifact_type="z3_constraint_pattern",
                source=f"z3_solver:{problem_id}",
                metadata=metadata,
                confidence=0.8  # Default confidence for patterns
            )
            
            return artifact_id
            
        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")
            return None
    
    async def _store_strategy(
        self,
        strategy: Dict[str, Any],
        problem_id: str
    ) -> Optional[str]:
        """Store a solution strategy."""
        try:
            content = f"Strategy: {strategy.get('name', 'unknown')}\n"
            content += f"Pattern: {strategy.get('pattern', '')}\n"
            content += f"Tactics: {', '.join(strategy.get('tactics', []))}"
            
            metadata = {
                "strategy_id": strategy.get("id"),
                "problem_pattern": strategy.get("pattern"),
                "tactics": strategy.get("tactics", []),
                "success_rate": strategy.get("success_rate"),
                "avg_time": strategy.get("avg_time")
            }
            
            # Calculate confidence from success rate
            success_rate_str = strategy.get("success_rate", "0%")
            confidence = float(success_rate_str.rstrip("%")) / 100
            
            artifact_id = await self.storage.store_knowledge_artifact(
                content=content,
                artifact_type="z3_strategy",
                source=f"z3_solver:{problem_id}",
                metadata=metadata,
                confidence=confidence
            )
            
            return artifact_id
            
        except Exception as e:
            logger.error(f"Failed to store strategy: {e}")
            return None
    
    # ========================================================================
    # Knowledge Retrieval and Recommendations
    # ========================================================================
    
    async def get_recommended_strategy(
        self,
        problem_features: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get a recommended strategy for a problem.
        
        Args:
            problem_features: Characteristics of the problem
            
        Returns:
            Recommended strategy or None
        """
        if not self.z3_extractor:
            return None
        
        strategy = self.z3_extractor.recommend_strategy(problem_features)
        return strategy.to_dict() if strategy else None
    
    async def search_similar_patterns(
        self,
        query: str,
        pattern_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar patterns in stored knowledge.
        
        Args:
            query: Search query
            pattern_type: Optional type filter
            top_k: Number of results
            
        Returns:
            List of matching patterns
        """
        if not self.storage:
            return []
        
        try:
            # Search for Z3-related artifacts
            artifact_type = f"z3_{pattern_type}" if pattern_type else None
            
            results = await self.storage.search_knowledge_artifacts(
                query=query,
                artifact_type=artifact_type,
                top_k=top_k
            )
            
            return [
                {
                    "id": r.id,
                    "type": r.artifact_type,
                    "content": r.content,
                    "metadata": r.metadata,
                    "confidence": r.confidence,
                    "created_at": r.created_at.isoformat()
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error({
                "msg": "Failed to search patterns",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return []
    
    async def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of Z3 knowledge in the system."""
        summary = {
            "z3_extractor": self.z3_extractor.get_knowledge_summary() if self.z3_extractor else None,
            "extraction_stats": {
                "total_extractions": self.extraction_count,
                "total_stored": self.storage_count
            },
            "storage_available": self.storage is not None
        }
        
        return summary
    
    # ========================================================================
    # Integration Helpers
    # ========================================================================
    
    async def process_solver_result(
        self,
        result: Any,
        problem_statement: str,
        problem_id: Optional[str] = None,
        problem_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Complete pipeline: extract and store knowledge from solver result.
        
        Args:
            result: Z3 solver result
            problem_statement: Original problem
            problem_id: Optional problem identifier
            problem_type: Problem classification
            
        Returns:
            Processing results
        """
        problem_id = problem_id or str(uuid.uuid4())
        
        # Extract knowledge
        extracted = await self.extract_from_solver_result(
            result, problem_statement, problem_type
        )
        
        # Store knowledge
        artifact_ids = await self.store_extracted_knowledge(extracted, problem_id)
        
        return {
            "problem_id": problem_id,
            "extracted": extracted,
            "stored_artifacts": artifact_ids,
            "success": len(artifact_ids) > 0 or not extracted.get("error")
        }
    
    async def close(self):
        """Close storage connections."""
        if self.storage:
            await self.storage.close()
            logger.info("Z3 knowledge integration closed")


# =============================================================================
# Global Instance
# =============================================================================

_z3_knowledge_integration: Optional[Z3KnowledgeIntegration] = None


async def get_z3_knowledge_integration(
    storage_engine: Optional[Any] = None
) -> 'Z3KnowledgeIntegration':
    """Get global Z3 knowledge integration instance."""
    global _z3_knowledge_integration
    if _z3_knowledge_integration is None:
        _z3_knowledge_integration = Z3KnowledgeIntegration(storage_engine)
        await _z3_knowledge_integration.initialize()
    return _z3_knowledge_integration


# =============================================================================
# Auto-Extraction Hook
# =============================================================================

class Z3KnowledgeExtractionHook:
    """
    Hook for automatic knowledge extraction from Z3 operations.
    Can be registered with Z3 solver to auto-extract knowledge.
    """
    
    def __init__(self, integration: Optional[Z3KnowledgeIntegration] = None):
        self.integration = integration
        self.enabled = True
    
    async def on_solver_result(
        self,
        result: Any,
        problem: str,
        problem_type: str = "general"
    ):
        """Called when a solver result is available."""
        if not self.enabled or not self.integration:
            return
        
        try:
            await self.integration.process_solver_result(
                result=result,
                problem_statement=problem,
                problem_type=problem_type
            )
        except Exception as e:
            logger.error(f"Auto-extraction failed: {e}")


# =============================================================================
# Example Usage
# =============================================================================

async def example_integration():
    """Example: Z3 knowledge integration."""
    print("Z3 Knowledge Integration Example")
    print("=" * 50)
    
    # Initialize integration
    integration = await get_z3_knowledge_integration()
    
    # Create a mock solver result
    class MockResult:
        success = True
        model = type('Model', (), {'assignments': {'x': 5, 'y': 10}})()
        constraints = ["(> x 0)", "(< x 10)", "(= y (+ x 5))"]
        solving_time = 1.5
        tactics_used = ["simplify", "solve-eqs", "smt"]
        config = {"timeout": 30}
    
    result = MockResult()
    
    # Process the result
    processing = await integration.process_solver_result(
        result=result,
        problem_statement="Find x and y satisfying the constraints",
        problem_type="linear"
    )
    
    print(f"\nProcessing result:")
    print(f"  Problem ID: {processing['problem_id']}")
    print(f"  Insights extracted: {len(processing['extracted']['insights'])}")
    print(f"  Patterns extracted: {len(processing['extracted']['patterns'])}")
    print(f"  Strategies extracted: {len(processing['extracted']['strategies'])}")
    print(f"  Artifacts stored: {len(processing['stored_artifacts'])}")
    
    # Get knowledge summary
    summary = await integration.get_knowledge_summary()
    print(f"\nKnowledge Summary:")
    print(f"  Total extractions: {summary['extraction_stats']['total_extractions']}")
    print(f"  Storage available: {summary['storage_available']}")
    
    # Close
    await integration.close()


if __name__ == "__main__":
    asyncio.run(example_integration())

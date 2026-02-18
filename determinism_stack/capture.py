"""
Evolved Code Capture - Persistence for deterministic code evolution.

Following Federation Constitution:
- Law of Idempotency: Duplicate captures are safe.
- Law of UTC: ISO-8601 timestamps.
- Logic: Integrates with KnowledgeEngine for storage.
"""

import os
import time
import uuid
import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from .deps import optional_import

logger = logging.getLogger(__name__)

class EvolvedCodeCapturer:
    """Python implementation of the Evolved Code Capture system."""

    def __init__(self):
        self._ke_module = optional_import("knowledge_engine")
        self._engine = None
        self._initialized = False

    def _ensure_engine(self):
        if self._initialized:
            return
        if not self._ke_module:
            return
        
        try:
            # Try to get the integrated engine
            for attr in ["IntegratedKnowledgeEngine", "KnowledgeEngine", "create_knowledge_engine"]:
                entry = getattr(self._ke_module, attr, None)
                if entry:
                    if attr == "create_knowledge_engine":
                        # Simplification: we don't run async here for init if we can avoid it
                        pass 
                    elif callable(entry):
                        self._engine = entry()
                    break
            self._initialized = True
        except Exception as exc:
            logger.debug(f"Failed to initialize KnowledgeEngine for capture: {exc}")

    def capture_evolution(
        self,
        problem: str,
        solution: str,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture evolved code and store it in knowledge systems.
        
        Args:
            problem: The original problem statement.
            solution: The evolved code solution.
            metrics: Evolution metrics (fitness, etc.)
            metadata: Additional metadata.
            
        Returns:
            Result of the capture operation.
        """
        self._ensure_engine()
        
        capture_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        artifact = {
            "id": capture_id,
            "type": "evolved_code",
            "problem": problem,
            "content": solution,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "timestamp": timestamp,
            "correlation_id": metadata.get("correlation_id") if metadata else None
        }
        
        # Store in Knowledge Engine if available
        storage_status = "not_available"
        if self._engine:
            try:
                # Real storage call - using engine.store_artifact
                if hasattr(self._engine, "store_artifact"):
                    # Create KnowledgeArtifact
                    try:
                        from knowledge_engine import KnowledgeArtifact
                        art = KnowledgeArtifact(
                            artifact_id=capture_id,
                            artifact_type="evolved_code",
                            source_system="determinism_stack",
                            content=json.dumps(artifact),
                            confidence=1.0,
                            metadata=metadata or {}
                        )
                        # Store it
                        self._engine.store_artifact(art)
                        storage_status = "success"
                    except ImportError:
                        # Fallback if KnowledgeArtifact class not directly importable
                        self._engine.store_artifact(artifact)
                        storage_status = "success"
                else:
                    # File-based backup if engine doesn't support store_artifact
                    storage_status = self._store_to_file(artifact)
            except Exception as exc:
                logger.error(f"Failed to store evolved code: {exc}")
                storage_status = "failed"
        else:
            # Automatic file-based backup if engine is missing
            storage_status = self._store_to_file(artifact)
        
        logger.info(f"Captured evolved code: {capture_id} (Status: {storage_status})")
        
        return {
            "success": storage_status == "success",
            "capture_id": capture_id,
            "timestamp": timestamp,
            "storage": storage_status
        }

    def _store_to_file(self, artifact: Dict[str, Any]) -> str:
        """Backup storage method when knowledge engine is unavailable."""
        try:
            path = Path("artifacts/evolved_code")
            path.mkdir(parents=True, exist_ok=True)
            filename = f"code_{artifact['id'][:8]}_{int(time.time())}.json"
            with open(path / filename, "w") as f:
                json.dump(artifact, f, indent=2)
            return "success"
        except Exception as exc:
            logger.error(f"File storage failed: {exc}")
            return "failed"

    def search_similar(self, problem: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar previously evolved solutions."""
        self._ensure_engine()
        
        # 1. Try Knowledge Engine first
        if self._engine and hasattr(self._engine, "query"):
            try:
                # Use engine's semantic search
                results = self._engine.query(problem, types=["evolved_code"], limit=limit)
                if results:
                    return results
            except Exception as exc:
                logger.debug(f"Knowledge Engine query failed: {exc}")
        
        # 2. Heuristic fallback: Search local artifacts
        try:
            path = Path("artifacts/evolved_code")
            if not path.exists():
                return []
                
            from .utils import similarity
            candidates = []
            for file in path.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                    score = similarity(problem, data.get("problem", ""))
                    if score > 0.5:
                        candidates.append((score, data))
            
            # Sort by similarity and return top N
            candidates.sort(key=lambda x: x[0], reverse=True)
            return [c[1] for c in candidates[:limit]]
        except Exception:
            pass
            
        return []

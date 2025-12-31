"""
Hybrid Knowledge Manager

Bridges RAGBits real-time storage with the existing structured knowledge base.
Provides unified access to both semantic search and structured queries.
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HybridKnowledgeManager:
    """
    Unified knowledge management combining RAGBits and existing KB.

    This manager provides:
    - Fast semantic search via RAGBits vector store
    - Structured queries via existing knowledge base
    - Automatic synchronization between both systems
    - Unified query interface

    Usage:
        hybrid = HybridKnowledgeManager(
            ragbits_store=document_search,
            existing_kb=knowledge_base
        )

        # Store artifact (goes to both systems)
        await hybrid.store_artifact(
            artifact={"content": "...", "metadata": {...}},
            stage="stage_3"
        )

        # Retrieve context (combines both sources)
        context = await hybrid.retrieve_context(
            query="user authentication",
            filters={"type": "solution"}
        )
    """

    def __init__(self, ragbits_store, existing_kb=None):
        """
        Initialize the hybrid knowledge manager.

        Args:
            ragbits_store: RAGBits DocumentSearch instance
            existing_kb: Optional existing knowledge base instance
        """
        self.ragbits = ragbits_store
        self.kb = existing_kb
        self.sync_enabled = existing_kb is not None

        logger.info(
            f"HybridKnowledgeManager initialized "
            f"(sync_enabled: {self.sync_enabled})"
        )

    async def store_artifact(
        self,
        artifact: Dict[str, Any],
        stage: str,
        sync_to_kb: bool = True
    ) -> Dict[str, str]:
        """
        Store an artifact in both RAGBits and existing KB.

        Args:
            artifact: Artifact dict with 'content' and 'metadata'
            stage: Workflow stage identifier
            sync_to_kb: Whether to sync to existing KB

        Returns:
            Dict with 'ragbits_id' and 'kb_id' if applicable

        Example:
            >>> result = await hybrid.store_artifact(
            ...     artifact={
            ...         "content": "Implement JWT authentication...",
            ...         "metadata": {"team": "blue", "sub_problem_id": "sub_1"}
            ...     },
            ...     stage="stage_3"
            ... )
            >>> print(result)
            {'ragbits_id': 'solution_draft_1234567890', 'kb_id': 'kb_98765'}
        """
        result = {"ragbits_id": None, "kb_id": None}

        # Store in RAGBits for immediate semantic search
        try:
            ragbits_id = await self._store_in_ragbits(artifact, stage)
            result["ragbits_id"] = ragbits_id
            logger.info(f"Stored in RAGBits: {ragbits_id}")
        except Exception as e:
            logger.error(f"Failed to store in RAGBits: {e}")

        # Sync to existing KB if enabled
        if sync_to_kb and self.sync_enabled and self.kb:
            try:
                kb_id = await self._store_in_kb(artifact, stage)
                result["kb_id"] = kb_id
                logger.info(f"Stored in KB: {kb_id}")
            except Exception as e:
                logger.error(f"Failed to store in KB: {e}")

        return result

    async def _store_in_ragbits(
        self,
        artifact: Dict[str, Any],
        stage: str
    ) -> str:
        """Store artifact in RAGBits vector store"""
        content = artifact.get("content", "")
        metadata = artifact.get("metadata", {})
        metadata["stage"] = stage

        # Generate artifact ID
        import time
        artifact_type = metadata.get("type", "unknown")
        artifact_id = f"{artifact_type}_{int(time.time() * 1000)}"
        metadata["artifact_id"] = artifact_id

        # Store in vector store
        await self.ragbits.ingest_text(text=content, metadata=metadata)

        return artifact_id

    async def _store_in_kb(
        self,
        artifact: Dict[str, Any],
        stage: str
    ) -> Optional[str]:
        """Store artifact in existing knowledge base"""
        if not self.kb:
            return None

        # Check if KB has store_artifact method
        if hasattr(self.kb, 'store_artifact'):
            return await self.kb.store_artifact(artifact, stage)
        elif hasattr(self.kb, 'add'):
            return await self.kb.add(artifact, stage)
        else:
            logger.warning("KB does not have standard store method")
            return None

    async def retrieve_context(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve context from both RAGBits and existing KB.

        Combines semantic search results from RAGBits with
        structured query results from existing KB.

        Args:
            query: Search query
            filters: Optional metadata filters
            top_k: Number of results from each source

        Returns:
            Dict with 'semantic' (RAGBits) and 'structured' (KB) results

        Example:
            >>> context = await hybrid.retrieve_context(
            ...     query="authentication best practices",
            ...     filters={"type": "solution"},
            ...     top_k=5
            ... )
            >>> semantic_results = context["semantic"]
            >>> structured_results = context["structured"]
        """
        context = {
            "query": query,
            "filters": filters,
            "semantic": [],
            "structured": [],
            "combined": []
        }

        # Semantic search from RAGBits
        try:
            ragbits_results = await self.ragbits.search(
                query=query,
                filters=filters,
                top_k=top_k
            )

            context["semantic"] = [
                {
                    "content": result.text_representation,
                    "metadata": result.metadata,
                    "similarity": result.metadata.get("similarity", 0),
                    "source": "ragbits"
                }
                for result in ragbits_results
            ]

            logger.info(f"Found {len(context['semantic'])} semantic results")

        except Exception as e:
            logger.error(f"RAGBits search failed: {e}")

        # Structured query from existing KB
        if self.sync_enabled and self.kb:
            try:
                kb_results = await self._query_kb(query, filters, top_k)
                context["structured"] = kb_results
                logger.info(f"Found {len(kb_results)} structured results")
            except Exception as e:
                logger.error(f"KB query failed: {e}")

        # Combine results (interleave by relevance)
        context["combined"] = self._combine_results(
            context["semantic"],
            context["structured"],
            top_k
        )

        return context

    async def _query_kb(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query the existing knowledge base"""
        if not self.kb:
            return []

        # Try different query methods
        if hasattr(self.kb, 'query'):
            return await self.kb.query(query, filters, top_k)
        elif hasattr(self.kb, 'search'):
            return await self.kb.search(query, filters, top_k)
        elif hasattr(self.kb, 'get'):
            # Fallback to get all
            results = await self.kb.get(filters)
            return results[:top_k] if results else []
        else:
            logger.warning("KB does not have standard query method")
            return []

    def _combine_results(
        self,
        semantic: List[Dict[str, Any]],
        structured: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic and structured results.

        Interleaves results by relevance score to provide
        a unified view from both sources.
        """
        combined = []

        # Add all semantic results
        for item in semantic:
            item_copy = item.copy()
            item_copy["relevance_type"] = "semantic"
            combined.append(item_copy)

        # Add all structured results
        for item in structured:
            item_copy = item.copy()
            item_copy["relevance_type"] = "structured"
            combined.append(item_copy)

        # Sort by similarity/relevance score
        combined.sort(
            key=lambda x: x.get("similarity", x.get("relevance", 0)),
            reverse=True
        )

        return combined[:top_k]

    async def retrieve_similar_solutions(
        self,
        problem_description: str,
        top_k: int = 5,
        min_success_rate: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar solutions from both knowledge sources.

        Args:
            problem_description: Problem to find solutions for
            top_k: Number of results
            min_success_rate: Minimum success rate threshold

        Returns:
            List of similar solutions
        """
        # Build filters
        filters = {
            "type": "solution_draft",
            "status": "final"
        }

        # Query both sources
        context = await self.retrieve_context(
            query=problem_description,
            filters=filters,
            top_k=top_k * 2  # Get more to filter
        )

        # Filter by success rate and combine
        all_solutions = context["combined"]

        # Filter by success rate
        filtered = [
            sol for sol in all_solutions
            if sol.get("metadata", {}).get("success_rate", 0) >= min_success_rate
        ]

        return filtered[:top_k]

    async def sync_artifact(
        self,
        artifact_id: str,
        source: str = "ragbits",
        destination: str = "kb"
    ) -> bool:
        """
        Sync an artifact from one system to another.

        Args:
            artifact_id: Artifact identifier
            source: Source system ("ragbits" or "kb")
            destination: Destination system ("ragbits" or "kb")

        Returns:
            True if sync successful
        """
        logger.info(f"Syncing artifact {artifact_id} from {source} to {destination}")

        if source == "ragbits" and destination == "kb":
            # Retrieve from RAGBits
            try:
                results = await self.ragbits.search(
                    query=artifact_id,
                    filters={"artifact_id": artifact_id},
                    top_k=1
                )

                if results:
                    artifact = {
                        "content": results[0].text_representation,
                        "metadata": results[0].metadata
                    }

                    # Store in KB
                    await self._store_in_kb(artifact, artifact["metadata"].get("stage", "unknown"))
                    return True

            except Exception as e:
                logger.error(f"Failed to sync from RAGBits to KB: {e}")

        elif source == "kb" and destination == "ragbits":
            # Implement KB to RAGBits sync if needed
            pass

        return False

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from both knowledge sources.

        Returns:
            Statistics dict with counts and metrics from both systems
        """
        stats = {
            "ragbits": {},
            "kb": {},
            "combined": {}
        }

        # RAGBits stats
        try:
            # This would require aggregation queries
            # For now, provide basic structure
            stats["ragbits"]["status"] = "active"
        except Exception as e:
            logger.error(f"Failed to get RAGBits stats: {e}")
            stats["ragbits"]["status"] = "error"

        # KB stats
        if self.kb:
            try:
                if hasattr(self.kb, 'statistics'):
                    stats["kb"] = await self.kb.statistics()
                else:
                    stats["kb"]["status"] = "active"
            except Exception as e:
                logger.error(f"Failed to get KB stats: {e}")
                stats["kb"]["status"] = "error"

        # Combined stats
        stats["combined"]["sync_enabled"] = self.sync_enabled
        stats["combined"]["sources_active"] = sum([
            1 for s in [stats["ragbits"], stats["kb"]]
            if s.get("status") == "active"
        ])

        return stats

    async def bulk_store(
        self,
        artifacts: List[Dict[str, Any]],
        stage: str,
        sync_to_kb: bool = True
    ) -> List[Dict[str, str]]:
        """
        Store multiple artifacts in batch.

        Args:
            artifacts: List of artifacts to store
            stage: Workflow stage
            sync_to_kb: Whether to sync to KB

        Returns:
            List of result dicts for each artifact
        """
        results = []

        for artifact in artifacts:
            result = await self.store_artifact(artifact, stage, sync_to_kb)
            results.append(result)

        logger.info(f"Bulk stored {len(artifacts)} artifacts in stage {stage}")
        return results

    async def clear_stage(
        self,
        stage: str,
        clear_ragbits: bool = True,
        clear_kb: bool = False
    ) -> bool:
        """
        Clear all artifacts from a specific stage.

        Use with caution - this is a destructive operation.

        Args:
            stage: Stage to clear
            clear_ragbits: Whether to clear RAGBits
            clear_kb: Whether to clear KB

        Returns:
            True if successful
        """
        logger.warning(f"Clearing stage {stage} (ragbits: {clear_ragbits}, kb: {clear_kb})")

        if clear_ragbits:
            # Implement RAGBits stage clearing
            # This would require delete functionality
            pass

        if clear_kb and self.kb:
            # Implement KB clearing
            if hasattr(self.kb, 'clear_stage'):
                await self.kb.clear_stage(stage)

        return True

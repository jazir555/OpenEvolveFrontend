"""
Intermediary Storage Manager

Real-time storage and retrieval system for workflow artifacts during execution.
Stores all artifacts immediately as they're generated, making them searchable
for later stages. Maintains versioning, linking, and lifecycle management.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


class ArtifactMetadata:
    """Metadata for a stored artifact"""

    def __init__(
        self,
        artifact_id: str,
        artifact_type: str,
        stage: str,
        team: Optional[str] = None,
        sub_problem_id: Optional[str] = None,
        status: str = "draft",
        links_to: Optional[List[str]] = None,
        **extra_metadata
    ):
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.stage = stage
        self.team = team
        self.sub_problem_id = sub_problem_id
        self.status = status  # draft → pending → verified → final
        self.links_to = links_to or []
        self.timestamp = datetime.utcnow().timestamp()
        self.extra_metadata = extra_metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "artifact_id": self.artifact_id,
            "type": self.artifact_type,
            "stage": self.stage,
            "team": self.team,
            "sub_problem_id": self.sub_problem_id,
            "status": self.status,
            "links_to": self.links_to,
            "timestamp": self.timestamp,
            **self.extra_metadata
        }


class IntermediaryStorageManager:
    """
    Real-time intermediary storage system for workflow artifacts.

    Manages the storage, retrieval, and lifecycle of artifacts generated
    during workflow execution. All artifacts are immediately indexed for
    semantic search and maintain relationships through linking.

    Usage:
        storage = IntermediaryStorageManager(document_search)

        # Store an artifact
        artifact_id = await storage.store_artifact(
            artifact_type="solution_draft",
            content="Solution content here...",
            metadata={"team": "blue", "sub_problem_id": "sub_1"}
        )

        # Retrieve context for a stage
        context = await storage.retrieve_context_for_stage(
            stage="stage_3_red_team_critique",
            sub_problem_id="sub_1"
        )
    """

    # Artifact lifecycle states
    STATUS_DRAFT = "draft"
    STATUS_PENDING = "pending"
    STATUS_VERIFIED = "verified"
    STATUS_FINAL = "final"

    # Artifact types
    TYPE_CONTENT_ANALYSIS = "content_analysis"
    TYPE_DECOMPOSITION_PLAN = "decomposition_plan"
    TYPE_SOLUTION_DRAFT = "solution_draft"
    TYPE_CRITIQUE = "critique"
    TYPE_VERIFICATION = "verification"
    TYPE_ASSEMBLED_SOLUTION = "assembled_solution"
    TYPE_FINAL_VERIFICATION = "final_verification"

    def __init__(self, document_search):
        """
        Initialize the storage manager.

        Args:
            document_search: RAGBits DocumentSearch instance for vector storage
        """
        self.document_search = document_search
        self._artifact_cache = {}  # In-memory cache for quick access
        self._artifact_counter = 0  # Counter for ensuring unique IDs
        logger.info("IntermediaryStorageManager initialized")

    async def store_artifact(
        self,
        artifact_type: str,
        content: str,
        metadata: Dict[str, Any],
        links_to: Optional[List[str]] = None,
        cache: bool = True
    ) -> str:
        """
        Store an artifact immediately with indexing.

        Args:
            artifact_type: Type of artifact (solution_draft, critique, etc.)
            content: The artifact content
            metadata: Additional metadata (stage, team, sub_problem_id, etc.)
            links_to: IDs of related artifacts to link with
            cache: Whether to cache in memory for faster retrieval

        Returns:
            artifact_id: Unique identifier for stored artifact

        Example:
            >>> artifact_id = await storage.store_artifact(
            ...     artifact_type="solution_draft",
            ...     content="Implement microservices architecture...",
            ...     metadata={
            ...         "stage": "stage_3",
            ...         "team": "blue",
            ...         "sub_problem_id": "sub_1"
            ...     }
            ... )
        """
        # Generate unique artifact ID
        self._artifact_counter += 1
        timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
        artifact_id = f"{artifact_type}_{timestamp_ms}_{self._artifact_counter}"

        # Create artifact metadata
        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            stage=metadata.get("stage", "unknown"),
            team=metadata.get("team"),
            sub_problem_id=metadata.get("sub_problem_id"),
            status=metadata.get("status", self.STATUS_DRAFT),
            links_to=links_to or [],
            **{k: v for k, v in metadata.items()
               if k not in ["stage", "team", "sub_problem_id", "status", "artifact_id", "type", "links_to"]}
        )

        # Prepare full metadata for storage
        full_metadata = artifact_metadata.to_dict()

        # Store in vector store (immediately indexed for semantic search)
        try:
            await self._ingest_to_vector_store(content, full_metadata)

            # Cache in memory if enabled
            if cache:
                self._artifact_cache[artifact_id] = {
                    "content": content,
                    "metadata": full_metadata
                }

            logger.info(
                f"Stored artifact {artifact_id} of type {artifact_type} "
                f"(stage: {artifact_metadata.stage}, "
                f"team: {artifact_metadata.team}, "
                f"links: {len(links_to or [])})"
            )

            return artifact_id

        except Exception as e:
            logger.error(f"Failed to store artifact {artifact_id}: {e}")
            raise

    async def _ingest_to_vector_store(
        self,
        text: str,
        metadata: Dict[str, Any]
    ):
        """
        Ingest text into the vector store.

        This method handles the actual storage in RAGBits DocumentSearch.
        Can be overridden for custom storage backends.
        """
        # Check if document_search has ingest_text method
        if hasattr(self.document_search, 'ingest_text'):
            await self.document_search.ingest_text(text=text, metadata=metadata)
        elif hasattr(self.document_search, 'ingest'):
            # Fallback for different RAGBits versions
            await self.document_search.ingest(text, metadata=metadata)
        else:
            raise AttributeError(
                "DocumentSearch instance does not have ingest_text or ingest method. "
                "Please check RAGBits version compatibility."
            )

    async def retrieve_artifact(
        self,
        artifact_id: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific artifact by ID.

        Args:
            artifact_id: Unique artifact identifier
            use_cache: Whether to check cache first

        Returns:
            Artifact dict with 'content' and 'metadata', or None if not found
        """
        # Check cache first
        if use_cache and artifact_id in self._artifact_cache:
            return self._artifact_cache[artifact_id]

        # Search in vector store
        try:
            results = await self.document_search.search(
                query=artifact_id,
                filters={"artifact_id": artifact_id},
                top_k=1
            )

            if results and len(results) > 0:
                result = results[0]
                artifact = {
                    "content": result.text_representation,
                    "metadata": result.metadata
                }
                # Cache for future access
                self._artifact_cache[artifact_id] = artifact
                return artifact

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
            return None

    async def retrieve_context_for_stage(
        self,
        stage: str,
        sub_problem_id: Optional[str] = None,
        query: Optional[str] = None,
        team: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Gather all relevant context for a workflow stage.

        This is how agents access artifacts from previous stages.
        Automatically retrieves the appropriate artifacts based on stage.

        Args:
            stage: The workflow stage (e.g., "stage_3_red_team_critique")
            sub_problem_id: Optional sub-problem identifier
            query: Optional semantic query for historical artifacts
            team: Optional team filter

        Returns:
            Context dict with 'artifacts', 'similar_historical', and 'stage' keys

        Example:
            >>> context = await storage.retrieve_context_for_stage(
            ...     stage="stage_3_red_team_critique",
            ...     sub_problem_id="sub_1"
            ... )
            >>> blue_solution = context["artifacts"]["blue_solution"]
        """
        context = {
            "stage": stage,
            "artifacts": {},
            "similar_historical": [],
            "metadata": {
                "sub_problem_id": sub_problem_id,
                "team": team,
                "query": query
            }
        }

        try:
            # Stage-specific context gathering
            if "stage_3" in stage:
                await self._gather_stage_3_context(
                    context, stage, sub_problem_id, query, team
                )
            elif "stage_1" in stage:
                await self._gather_stage_1_context(context)
            elif "stage_4" in stage:
                await self._gather_stage_4_context(context, sub_problem_id)
            elif "stage_5" in stage:
                await self._gather_stage_5_context(context, sub_problem_id)

            logger.info(f"Gathered context for stage {stage}: "
                       f"{len(context['artifacts'])} artifact types, "
                       f"{len(context['similar_historical'])} historical items")

            return context

        except Exception as e:
            logger.error(f"Failed to gather context for stage {stage}: {e}")
            return context

    async def _gather_stage_3_context(
        self,
        context: Dict[str, Any],
        stage: str,
        sub_problem_id: Optional[str],
        query: Optional[str],
        team: Optional[str]
    ):
        """Gather context for Stage 3 (Sub-Problem Solving)"""
        # Get content analysis from Stage 0
        context["artifacts"]["content_analysis"] = await self._search_artifacts(
            query="content analysis",
            filters={"type": self.TYPE_CONTENT_ANALYSIS},
            top_k=1
        )

        # Get current decomposition plan from Stage 1
        context["artifacts"]["decomposition_plan"] = await self._search_artifacts(
            query="decomposition plan",
            filters={"type": self.TYPE_DECOMPOSITION_PLAN, "is_current": True},
            top_k=1
        )

        # Red Team needs Blue Team's solution
        if "red_team" in stage or "critique" in stage:
            if sub_problem_id:
                context["artifacts"]["blue_solution"] = await self._search_artifacts(
                    query=f"solution for {sub_problem_id}",
                    filters={
                        "type": self.TYPE_SOLUTION_DRAFT,
                        "sub_problem_id": sub_problem_id,
                        "team": "blue"
                    },
                    top_k=1
                )

                # Get similar critiques for reference
                if query:
                    context["similar_historical"] = await self._search_artifacts(
                        query=query,
                        filters={"type": self.TYPE_CRITIQUE},
                        top_k=3
                    )

        # Gold Team needs both solution and critique
        if "gold_team" in stage or "verification" in stage:
            if sub_problem_id:
                context["artifacts"]["solution"] = await self._search_artifacts(
                    query=f"solution for {sub_problem_id}",
                    filters={
                        "type": self.TYPE_SOLUTION_DRAFT,
                        "sub_problem_id": sub_problem_id
                    },
                    top_k=1
                )

                context["artifacts"]["critique"] = await self._search_artifacts(
                    query=f"critique of {sub_problem_id}",
                    filters={
                        "type": self.TYPE_CRITIQUE,
                        "sub_problem_id": sub_problem_id
                    },
                    top_k=1
                )

        # Get similar solutions from history for Blue Team
        if query and ("blue_team" in stage or "solution" in stage):
            context["similar_historical"] = await self._search_artifacts(
                query=query,
                filters={
                    "type": self.TYPE_SOLUTION_DRAFT,
                    "status": self.STATUS_FINAL
                },
                top_k=5
            )

    async def _gather_stage_1_context(self, context: Dict[str, Any]):
        """Gather context for Stage 1 (Decomposition)"""
        context["artifacts"]["content_analysis"] = await self._search_artifacts(
            query="content analysis",
            filters={"type": self.TYPE_CONTENT_ANALYSIS},
            top_k=1
        )

    async def _gather_stage_4_context(
        self,
        context: Dict[str, Any],
        sub_problem_id: Optional[str]
    ):
        """Gather context for Stage 4 (Reassembly)"""
        # Get all verified sub-problem solutions
        context["artifacts"]["verified_solutions"] = await self._search_artifacts(
            query="verified solutions",
            filters={
                "type": self.TYPE_SOLUTION_DRAFT,
                "status": self.STATUS_VERIFIED
            },
            top_k=20
        )

    async def _gather_stage_5_context(
        self,
        context: Dict[str, Any],
        sub_problem_id: Optional[str]
    ):
        """Gather context for Stage 5 (Final Verification)"""
        # Get assembled solution
        context["artifacts"]["assembled_solution"] = await self._search_artifacts(
            query="assembled solution",
            filters={"type": self.TYPE_ASSEMBLED_SOLUTION},
            top_k=1
        )

    async def _search_artifacts(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for artifacts in the vector store.

        Args:
            query: Search query
            filters: Metadata filters
            top_k: Number of results to return

        Returns:
            List of artifact dicts with 'content' and 'metadata'
        """
        try:
            results = await self.document_search.search(
                query=query,
                filters=filters,
                top_k=top_k
            )

            return [
                {
                    "content": result.text_representation,
                    "metadata": result.metadata
                }
                for result in results
            ]

        except Exception as e:
            logger.error(f"Failed to search artifacts: {e}")
            return []

    async def update_artifact_status(
        self,
        artifact_id: str,
        new_status: str
    ) -> bool:
        """
        Update artifact lifecycle status.

        Status flow: draft → pending → verified → final

        Args:
            artifact_id: Artifact to update
            new_status: New status value

        Returns:
            True if successful, False otherwise
        """
        # Retrieve current artifact
        artifact = await self.retrieve_artifact(artifact_id, use_cache=False)

        if not artifact:
            logger.warning(f"Artifact {artifact_id} not found for status update")
            return False

        # Create new version with updated status
        new_metadata = artifact["metadata"].copy()
        new_metadata["status"] = new_status
        new_metadata["previous_version"] = artifact_id
        new_metadata["status_updated_at"] = datetime.utcnow().timestamp()

        # Store updated version
        try:
            await self.store_artifact(
                artifact_type=new_metadata.get("type", "unknown"),
                content=artifact["content"],
                metadata=new_metadata,
                links_to=new_metadata.get("links_to", []),
                cache=True
            )

            logger.info(f"Updated artifact {artifact_id} status to {new_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update artifact status: {e}")
            return False

    async def get_artifact_chain(
        self,
        artifact_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve full chain of related artifacts.

        Follows links to trace relationships between artifacts.

        Example: solution → critique → verification → refined_solution

        Args:
            artifact_id: Starting artifact ID

        Returns:
            List of artifacts with their position in the chain
        """
        chain = []
        visited = set()

        await self._build_chain_recursive(artifact_id, chain, visited, position="root")

        return chain

    async def _build_chain_recursive(
        self,
        artifact_id: str,
        chain: List[Dict[str, Any]],
        visited: set,
        position: str = "linked"
    ):
        """Recursively build artifact chain by following links"""
        if artifact_id in visited:
            return

        visited.add(artifact_id)

        artifact = await self.retrieve_artifact(artifact_id, use_cache=True)
        if not artifact:
            return

        chain.append({
            "artifact": artifact,
            "artifact_id": artifact_id,
            "position": position
        })

        # Follow linked artifacts
        links = artifact["metadata"].get("links_to", [])
        for link_id in links:
            await self._build_chain_recursive(link_id, chain, visited, "linked")

    async def rollback_to_version(
        self,
        artifact_id: str,
        target_version_id: str
    ) -> bool:
        """
        Rollback an artifact to a previous version.

        Args:
            artifact_id: Current artifact ID
            target_version_id: Target version to rollback to

        Returns:
            True if successful, False otherwise
        """
        # Retrieve target version
        target = await self.retrieve_artifact(target_version_id, use_cache=False)

        if not target:
            logger.warning(f"Target version {target_version_id} not found for rollback")
            return False

        # Create new version with content from target
        new_metadata = target["metadata"].copy()
        new_metadata["rolled_back_from"] = artifact_id
        new_metadata["rollback_timestamp"] = datetime.utcnow().timestamp()

        try:
            await self.store_artifact(
                artifact_type=new_metadata.get("type", "unknown"),
                content=target["content"],
                metadata=new_metadata,
                links_to=new_metadata.get("links_to", []),
                cache=True
            )

            logger.info(f"Rolled back artifact {artifact_id} to version {target_version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback artifact: {e}")
            return False

    async def get_artifacts_by_stage(
        self,
        stage: str,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all artifacts for a given stage.

        Args:
            stage: Stage identifier
            status: Optional status filter

        Returns:
            List of artifacts from the stage
        """
        filters = {"stage": stage}
        if status:
            filters["status"] = status

        return await self._search_artifacts(
            query=f"artifacts from stage {stage}",
            filters=filters,
            top_k=100
        )

    async def get_artifacts_by_sub_problem(
        self,
        sub_problem_id: str,
        artifact_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all artifacts for a given sub-problem.

        Args:
            sub_problem_id: Sub-problem identifier
            artifact_type: Optional artifact type filter

        Returns:
            List of artifacts for the sub-problem
        """
        filters = {"sub_problem_id": sub_problem_id}
        if artifact_type:
            filters["type"] = artifact_type

        return await self._search_artifacts(
            query=f"artifacts for sub-problem {sub_problem_id}",
            filters=filters,
            top_k=50
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the artifact cache.

        Returns:
            Cache statistics dict
        """
        return {
            "cached_artifacts": len(self._artifact_cache),
            "artifact_ids": list(self._artifact_cache.keys())
        }

    def clear_cache(self):
        """Clear the in-memory artifact cache."""
        self._artifact_cache.clear()
        logger.info("Artifact cache cleared")

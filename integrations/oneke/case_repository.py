"""
OneKE Case Repository

This module implements a case-based learning system for storing and
retrieving extraction cases to improve knowledge extraction quality.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from .case import Case, CaseSimilarity, CaseStatistics

logger = logging.getLogger(__name__)


class OneKECaseRepository:
    """
    Repository for storing and retrieving extraction cases.

    Features:
    - Case storage with metadata
    - Similarity-based retrieval using semantic search
    - Case quality tracking
    - Automatic case updates and persistence
    - Export/import functionality
    """

    def __init__(
        self,
        storage_path: str = "data/oneke_cases.json",
        embedding_model: str = None,
        auto_save: bool = True,
        save_interval: int = 100
    ):
        """
        Initialize the case repository.

        Args:
            storage_path: Path to case storage file
            embedding_model: Name of sentence transformer model
            auto_save: Whether to auto-save cases
            save_interval: Save every N cases
        """
        self.storage_path = Path(storage_path)
        self.embedding_model_name = embedding_model or "sentence-transformers/all-mpnet-base-v2"
        self.auto_save = auto_save
        self.save_interval = save_interval

        self.cases: List[Case] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_model = None
        self._save_counter = 0

        self.logger = logging.getLogger(f"{__name__}.OneKECaseRepository")

        # Create storage directory
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> bool:
        """
        Initialize the repository and load existing cases.

        Returns:
            True if initialization successful
        """
        try:
            # Load existing cases
            await self._load_cases()

            # Initialize embedding model
            await self._init_embedding_model()

            # Generate embeddings for loaded cases
            if self.cases:
                await self._generate_embeddings()

            self.logger.info(
                f"Case repository initialized with {len(self.cases)} cases"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize case repository: {e}")
            return False

    async def _init_embedding_model(self):
        """Initialize sentence transformer model for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info("Embedding model loaded successfully")

        except ImportError:
            self.logger.warning(
                "sentence-transformers not installed. "
                "Using fallback similarity (keyword-based)."
            )
            self.embedding_model = None

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    async def _load_cases(self):
        """Load cases from storage file."""
        if not self.storage_path.exists():
            self.logger.info(f"No existing case file found: {self.storage_path}")
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cases_data = data.get('cases', [])
            self.cases = [Case.from_dict(case_data) for case_data in cases_data]

            self.logger.info(f"Loaded {len(self.cases)} cases from {self.storage_path}")

        except Exception as e:
            self.logger.error(f"Failed to load cases: {e}")
            self.cases = []

    async def _save_cases(self):
        """Save cases to storage file."""
        try:
            data = {
                'version': '1.0',
                'saved_at': datetime.utcnow().isoformat(),
                'total_cases': len(self.cases),
                'cases': [case.to_dict() for case in self.cases]
            }

            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved {len(self.cases)} cases to {self.storage_path}")

        except Exception as e:
            self.logger.error(f"Failed to save cases: {e}")

    async def _generate_embeddings(self):
        """Generate embeddings for all cases."""
        if not self.embedding_model or not self.cases:
            return

        try:
            # Generate embeddings from input text
            texts = [case.input_text for case in self.cases]
            self.embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            self.logger.debug(f"Generated embeddings for {len(self.cases)} cases")

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            self.embeddings = None

    async def add_case(
        self,
        case: Case,
        quality_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a case to the repository.

        Args:
            case: Case to add
            quality_score: Optional quality score (overrides case.quality_score)
            metadata: Optional additional metadata
        """
        try:
            # Update quality if provided
            if quality_score is not None:
                case.update_quality(quality_score)

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    case.add_metadata(key, value)

            # Add to repository
            self.cases.append(case)

            # Regenerate embeddings
            if self.embedding_model:
                new_embedding = self.embedding_model.encode(
                    [case.input_text],
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                if self.embeddings is None:
                    self.embeddings = new_embedding
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embedding])

            # Auto-save if enabled
            if self.auto_save:
                self._save_counter += 1
                if self._save_counter >= self.save_interval:
                    await self._save_cases()
                    self._save_counter = 0

            self.logger.info(f"Added case {case.case_id} to repository")

        except Exception as e:
            self.logger.error(f"Failed to add case: {e}")

    async def retrieve_similar_cases(
        self,
        query: Dict[str, Any],
        top_k: int = 5,
        min_similarity: float = 0.7,
        domain: Optional[str] = None
    ) -> List[CaseSimilarity]:
        """
        Retrieve similar cases using semantic search.

        Args:
            query: Query dict with 'input_text' or 'text' field
            top_k: Number of cases to retrieve
            min_similarity: Minimum similarity threshold
            domain: Optional domain filter

        Returns:
            List of CaseSimilarity objects
        """
        try:
            if not self.cases:
                self.logger.warning("No cases in repository")
                return []

            # Get query text
            query_text = query.get('input_text') or query.get('text', '')
            if not query_text:
                self.logger.warning("No query text provided")
                return []

            # Filter by domain if specified
            candidate_cases = self.cases
            if domain:
                candidate_cases = [c for c in self.cases if c.domain == domain]

            if not candidate_cases:
                self.logger.warning(f"No cases found for domain: {domain}")
                return []

            # Compute similarities
            if self.embedding_model and self.embeddings is not None:
                similarities = await self._compute_semantic_similarity(
                    query_text,
                    candidate_cases
                )
            else:
                similarities = await self._compute_keyword_similarity(
                    query_text,
                    candidate_cases
                )

            # Filter and sort
            filtered = [
                (case, sim)
                for case, sim in similarities
                if sim >= min_similarity
            ]
            filtered.sort(key=lambda x: x[1], reverse=True)

            # Take top_k
            top_cases = filtered[:top_k]

            # Create CaseSimilarity objects
            results = []
            for case, similarity in top_cases:
                match_reasons = self._generate_match_reasons(query_text, case)
                results.append(CaseSimilarity(
                    case=case,
                    similarity=similarity,
                    match_reasons=match_reasons
                ))

            self.logger.info(
                f"Retrieved {len(results)} similar cases (threshold={min_similarity})"
            )
            return results

        except Exception as e:
            self.logger.error(f"Failed to retrieve similar cases: {e}")
            return []

    async def _compute_semantic_similarity(
        self,
        query_text: str,
        candidate_cases: List[Case]
    ) -> List[tuple[Case, float]]:
        """Compute semantic similarity using embeddings."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                [query_text],
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Get indices of candidate cases
            candidate_indices = [self.cases.index(c) for c in candidate_cases]

            # Compute cosine similarity
            candidate_embeddings = self.embeddings[candidate_indices]
            similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

            # Normalize to 0-1
            similarities = (similarities + 1) / 2

            return list(zip(candidate_cases, similarities))

        except Exception as e:
            self.logger.error(f"Failed to compute semantic similarity: {e}")
            return []

    async def _compute_keyword_similarity(
        self,
        query_text: str,
        candidate_cases: List[Case]
    ) -> List[tuple[Case, float]]:
        """Compute keyword-based similarity (fallback)."""
        query_words = set(query_text.lower().split())

        similarities = []
        for case in candidate_cases:
            case_words = set(case.input_text.lower().split())

            # Jaccard similarity
            intersection = query_words & case_words
            union = query_words | case_words
            similarity = len(intersection) / len(union) if union else 0.0

            similarities.append((case, similarity))

        return similarities

    def _generate_match_reasons(self, query_text: str, case: Case) -> List[str]:
        """Generate reasons why a case matches the query."""
        reasons = []

        # Domain match
        reasons.append(f"Same domain: {case.domain}")

        # Schema match
        if 'schema' in case.metadata:
            reasons.append(f"Similar schema: {case.schema}")

        # Quality indicator
        if case.quality_score >= 0.8:
            reasons.append(f"High quality case ({case.quality_score:.2f})")

        # Keyword overlap
        query_words = set(query_text.lower().split())
        case_words = set(case.input_text.lower().split())
        overlap = query_words & case_words

        if overlap:
            top_keywords = sorted(overlap, key=len, reverse=True)[:3]
            reasons.append(f"Shared keywords: {', '.join(top_keywords)}")

        return reasons

    async def get_good_cases(
        self,
        domain: str,
        min_quality: float = 0.8,
        limit: int = 10
    ) -> List[Case]:
        """Get high-quality cases for a domain."""
        cases = [
            c for c in self.cases
            if c.domain == domain and c.quality_score >= min_quality
        ]

        # Sort by quality
        cases.sort(key=lambda c: c.quality_score, reverse=True)

        return cases[:limit]

    async def get_bad_cases(
        self,
        domain: str,
        max_quality: float = 0.5,
        limit: int = 10
    ) -> List[Case]:
        """Get low-quality cases for analysis."""
        cases = [
            c for c in self.cases
            if c.domain == domain and c.quality_score <= max_quality
        ]

        # Sort by quality (ascending)
        cases.sort(key=lambda c: c.quality_score)

        return cases[:limit]

    async def update_case_quality(
        self,
        case_id: str,
        new_quality: float
    ):
        """Update quality score for a case."""
        for case in self.cases:
            if case.case_id == case_id:
                case.update_quality(new_quality)
                self.logger.info(f"Updated quality for case {case_id}: {new_quality}")

                if self.auto_save:
                    await self._save_cases()

                return

        self.logger.warning(f"Case {case_id} not found")

    async def get_statistics(self) -> CaseStatistics:
        """Get repository statistics."""
        if not self.cases:
            return CaseStatistics(
                total_cases=0,
                average_quality=0.0,
                domain_distribution={},
                quality_distribution={},
                recent_cases=[]
            )

        # Total cases
        total = len(self.cases)

        # Average quality
        avg_quality = sum(c.quality_score for c in self.cases) / total

        # Domain distribution
        domain_dist = {}
        for case in self.cases:
            domain_dist[case.domain] = domain_dist.get(case.domain, 0) + 1

        # Quality distribution
        quality_dist = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        }

        for case in self.cases:
            q = case.quality_score
            if q < 0.2:
                quality_dist['0.0-0.2'] += 1
            elif q < 0.4:
                quality_dist['0.2-0.4'] += 1
            elif q < 0.6:
                quality_dist['0.4-0.6'] += 1
            elif q < 0.8:
                quality_dist['0.6-0.8'] += 1
            else:
                quality_dist['0.8-1.0'] += 1

        # Recent cases
        recent = sorted(
            self.cases,
            key=lambda c: c.created_at,
            reverse=True
        )[:10]
        recent_ids = [c.case_id for c in recent]

        return CaseStatistics(
            total_cases=total,
            average_quality=avg_quality,
            domain_distribution=domain_dist,
            quality_distribution=quality_dist,
            recent_cases=recent_ids
        )

    async def export_cases(
        self,
        output_path: str,
        format: str = "json"
    ):
        """Export cases to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                data = {
                    'version': '1.0',
                    'exported_at': datetime.utcnow().isoformat(),
                    'total_cases': len(self.cases),
                    'cases': [case.to_dict() for case in self.cases]
                }

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Exported {len(self.cases)} cases to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export cases: {e}")

    async def import_cases(
        self,
        input_path: str,
        format: str = "json"
    ):
        """Import cases from file."""
        try:
            input_path = Path(input_path)

            if not input_path.exists():
                raise FileNotFoundError(f"File not found: {input_path}")

            if format == "json":
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                cases_data = data.get('cases', [])
                new_cases = [Case.from_dict(case_data) for case_data in cases_data]

                # Add cases (avoiding duplicates by case_id)
                existing_ids = {c.case_id for c in self.cases}
                for case in new_cases:
                    if case.case_id not in existing_ids:
                        self.cases.append(case)
                        existing_ids.add(case.case_id)

                # Regenerate embeddings
                if self.embedding_model:
                    await self._generate_embeddings()

                self.logger.info(f"Imported {len(new_cases)} cases from {input_path}")

            else:
                raise ValueError(f"Unsupported import format: {format}")

        except Exception as e:
            self.logger.error(f"Failed to import cases: {e}")

    async def save(self):
        """Force save cases to storage."""
        await self._save_cases()
        self._save_counter = 0

    async def close(self):
        """Close the repository and save pending changes."""
        if self.auto_save and self._save_counter > 0:
            await self._save_cases()

        self.logger.info("Case repository closed")

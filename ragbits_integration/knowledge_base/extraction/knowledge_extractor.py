"""
Knowledge Extractor

Automatically extracts structured knowledge from workflow artifacts
including patterns, best practices, lessons learned, and solutions.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class KnowledgeEntityType(Enum):
    """Types of knowledge entities"""
    SOLUTION_PATTERN = "solution_pattern"
    BEST_PRACTICE = "best_practice"
    LESSON_LEARNED = "lesson_learned"
    ANTI_PATTERN = "anti_pattern"
    TECHNIQUE = "technique"
    PRINCIPLE = "principle"
    REQUIREMENT = "requirement"
    CONSTRAINT = "constraint"
    ASSUMPTION = "assumption"
    DEPENDENCY = "dependency"


@dataclass
class KnowledgeEntity:
    """A single knowledge entity"""
    entity_type: KnowledgeEntityType
    content: str
    confidence: float  # 0-1
    source_artifact_id: str
    source_section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_entities: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entity_type": self.entity_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "source_artifact_id": self.source_artifact_id,
            "source_section": self.source_section,
            "metadata": self.metadata,
            "related_entities": self.related_entities,
            "tags": list(self.tags),
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntity":
        """Create from dictionary"""
        return cls(
            entity_type=KnowledgeEntityType(data["entity_type"]),
            content=data["content"],
            confidence=data["confidence"],
            source_artifact_id=data["source_artifact_id"],
            source_section=data.get("source_section"),
            metadata=data.get("metadata", {}),
            related_entities=data.get("related_entities", []),
            tags=set(data.get("tags", [])),
            timestamp=data.get("timestamp", datetime.utcnow().timestamp())
        )


@dataclass
class ExtractionResult:
    """Result of knowledge extraction"""
    artifact_id: str
    entities: List[KnowledgeEntity]
    extraction_summary: Dict[str, int]
    processing_time_ms: float
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "artifact_id": self.artifact_id,
            "entities": [e.to_dict() for e in self.entities],
            "extraction_summary": self.extraction_summary,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class KnowledgeExtractor:
    """
    Automatically extracts structured knowledge from artifacts.

    Uses pattern matching, NLP techniques, and LLM-based extraction
    to identify valuable knowledge entities.

    Usage:
        extractor = KnowledgeExtractor(crewai_client)

        result = await extractor.extract_from_artifact(
            artifact_id="art_123",
            content="Implement JWT authentication...",
            artifact_type="solution"
        )

        # Access extracted entities
        for entity in result.entities:
            print(f"{entity.entity_type.value}: {entity.content}")
    """

    # Patterns for identifying knowledge entities
    PATTERNS = {
        KnowledgeEntityType.SOLUTION_PATTERN: [
            r"pattern:\s*(.+?)(?:\.|\n)",
            r"approach:\s*(.+?)(?:\.|\n)",
            r"architecture:\s*(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.BEST_PRACTICE: [
            r"best practice:\s*(.+?)(?:\.|\n)",
            r"recommended:\s*(.+?)(?:\.|\n)",
            r"should\s+(.+?)(?:\.|\n)",
            r"ideally\s+(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.LESSON_LEARNED: [
            r"lesson\s+learned:\s*(.+?)(?:\.|\n)",
            r"learned\s+that\s+(.+?)(?:\.|\n)",
            r"found\s+that\s+(.+?)(?:\.|\n)",
            r"discovered:\s*(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.ANTI_PATTERN: [
            r"avoid:\s*(.+?)(?:\.|\n)",
            r"should\s+not\s+(.+?)(?:\.|\n)",
            r"don't\s+(.+?)(?:\.|\n)",
            r"never:\s*(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.TECHNIQUE: [
            r"technique:\s*(.+?)(?:\.|\n)",
            r"method:\s*(.+?)(?:\.|\n)",
            r"strategy:\s*(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.PRINCIPLE: [
            r"principle:\s*(.+?)(?:\.|\n)",
            r"follows?\s+the\s+(.+?)\s+principle(?:\.|\n)",
            r"based\s+on\s+(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.REQUIREMENT: [
            r"requirement:\s*(.+?)(?:\.|\n)",
            r"must\s+(.+?)(?:\.|\n)",
            r"shall\s+(.+?)(?:\.|\n)",
            r"required:\s*(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.CONSTRAINT: [
            r"constraint:\s*(.+?)(?:\.|\n)",
            r"limited\s+by\s+(.+?)(?:\.|\n)",
            r"restricted\s+to\s+(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.ASSUMPTION: [
            r"assume[sd]?\s+(.+?)(?:\.|\n)",
            r"assuming\s+(.+?)(?:\.|\n)",
            r"assumption:\s*(.+?)(?:\.|\n)"
        ],
        KnowledgeEntityType.DEPENDENCY: [
            r"depend[s]?\s+on\s+(.+?)(?:\.|\n)",
            r"requires?\s+(.+?)(?:\.|\n)",
            r"relies?\s+on\s+(.+?)(?:\.|\n)"
        ]
    }

    def __init__(self, crewai_client=None):
        """
        Initialize knowledge extractor.

        Args:
            crewai_client: Optional CREWAI client for LLM-based extraction
        """
        self.crewai_client = crewai_client

        # Extraction statistics
        self.extraction_stats = {
            "artifacts_processed": 0,
            "entities_extracted": 0,
            "by_type": {entity_type.value: 0 for entity_type in KnowledgeEntityType}
        }

        logger.info("KnowledgeExtractor initialized")

    async def extract_from_artifact(
        self,
        artifact_id: str,
        content: str,
        artifact_type: str,
        use_llm: bool = True,
        min_confidence: float = 0.3
    ) -> ExtractionResult:
        """
        Extract knowledge from an artifact.

        Args:
            artifact_id: Artifact ID
            content: Artifact content
            artifact_type: Type of artifact
            use_llm: Whether to use LLM for extraction
            min_confidence: Minimum confidence threshold

        Returns:
            Extraction result
        """
        import time
        start_time = time.time()

        logger.info(f"Extracting knowledge from {artifact_id}")

        entities = []

        # Pattern-based extraction
        pattern_entities = await self._extract_by_patterns(
            content,
            artifact_id,
            min_confidence
        )
        entities.extend(pattern_entities)

        # LLM-based extraction (if enabled)
        if use_llm and self.crewai_client:
            llm_entities = await self._extract_by_llm(
                content,
                artifact_id,
                artifact_type
            )
            entities.extend(llm_entities)

        # Deduplicate and filter
        entities = await self._deduplicate_entities(entities)
        entities = [e for e in entities if e.confidence >= min_confidence]

        # Link related entities
        await self._link_related_entities(entities)

        # Generate summary
        extraction_summary = self._generate_summary(entities)

        processing_time = (time.time() - start_time) * 1000

        # Update statistics
        self.extraction_stats["artifacts_processed"] += 1
        self.extraction_stats["entities_extracted"] += len(entities)
        for entity in entities:
            self.extraction_stats["by_type"][entity.entity_type.value] += 1

        result = ExtractionResult(
            artifact_id=artifact_id,
            entities=entities,
            extraction_summary=extraction_summary,
            processing_time_ms=processing_time,
            metadata={
                "artifact_type": artifact_type,
                "content_length": len(content),
                "use_llm": use_llm
            }
        )

        logger.info(
            f"Extracted {len(entities)} entities from {artifact_id} "
            f"in {processing_time:.0f}ms"
        )

        return result

    async def _extract_by_patterns(
        self,
        content: str,
        artifact_id: str,
        min_confidence: float
    ) -> List[KnowledgeEntity]:
        """Extract entities using pattern matching"""
        entities = []
        content_lower = content.lower()

        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, content_lower, re.IGNORECASE)

                    for match in matches:
                        entity_content = match.group(1).strip()

                        # Filter out very short matches
                        if len(entity_content) < 10:
                            continue

                        # Calculate confidence based on context
                        confidence = self._calculate_pattern_confidence(
                            match,
                            content_lower
                        )

                        if confidence >= min_confidence:
                            # Extract tags from content
                            tags = self._extract_tags(entity_content)

                            entities.append(KnowledgeEntity(
                                entity_type=entity_type,
                                content=entity_content.capitalize(),
                                confidence=confidence,
                                source_artifact_id=artifact_id,
                                tags=tags
                            ))

                except re.error:
                    continue

        return entities

    async def _extract_by_llm(
        self,
        content: str,
        artifact_id: str,
        artifact_type: str
    ) -> List[KnowledgeEntity]:
        """Extract entities using LLM"""
        if not self.crewai_client:
            return []

        try:
            prompt = f"""Extract structured knowledge from the following {artifact_type}.

Identify and extract:
1. Solution patterns and approaches
2. Best practices and recommendations
3. Lessons learned
4. Techniques and methods
5. Requirements and constraints
6. Dependencies and assumptions

Content:
{content[:2000]}

Format as JSON:
{{
    "entities": [
        {{
            "type": "solution_pattern|best_practice|lesson_learned|technique|requirement|constraint|dependency|assumption",
            "content": "description",
            "confidence": 0.0-1.0,
            "tags": ["tag1", "tag2"]
        }}
    ]
}}"""

            response = await self.crewai_client.generate(
                prompt,
                temperature=0.3
            )

            # Parse LLM response
            entities = self._parse_llm_response(response, artifact_id)

            return entities

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

    def _parse_llm_response(
        self,
        response: Dict[str, Any],
        artifact_id: str
    ) -> List[KnowledgeEntity]:
        """Parse LLM response into entities"""
        entities = []

        try:
            response_text = response.get("text", "")

            # Try to extract JSON
            import json
            json_match = re.search(r'\{[\s\S]*\}', response_text)

            if json_match:
                data = json.loads(json_match.group())

                for entity_data in data.get("entities", []):
                    try:
                        entity_type = KnowledgeEntityType(
                            entity_data.get("type", "technique")
                        )

                        entities.append(KnowledgeEntity(
                            entity_type=entity_type,
                            content=entity_data.get("content", "")[:500],
                            confidence=entity_data.get("confidence", 0.7),
                            source_artifact_id=artifact_id,
                            tags=set(entity_data.get("tags", []))
                        ))
                    except (ValueError, KeyError):
                        continue

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")

        return entities

    def _calculate_pattern_confidence(
        self,
        match: re.Match,
        content: str
    ) -> float:
        """Calculate confidence score for pattern match"""
        base_confidence = 0.5

        # Boost confidence based on match length
        match_length = len(match.group(1))
        if match_length > 50:
            base_confidence += 0.2
        elif match_length > 30:
            base_confidence += 0.1

        # Check for contextual indicators
        context_start = max(0, match.start() - 50)
        context_end = min(len(content), match.end() + 50)
        context = content[context_start:context_end]

        # Look for technical indicators
        technical_terms = [
            "implement", "architecture", "design", "pattern",
            "system", "component", "service", "api"
        ]

        if any(term in context.lower() for term in technical_terms):
            base_confidence += 0.15

        # Look for completeness indicators
        if any(punct in context for punct in [".", "!", ";"]):
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _extract_tags(self, content: str) -> Set[str]:
        """Extract tags from content"""
        tags = set()

        # Common technical terms
        technical_keywords = [
            "api", "rest", "graphql", "jwt", "oauth",
            "microservices", "monolith", "serverless",
            "sql", "nosql", "database", "cache",
            "authentication", "authorization", "security",
            "performance", "scalability", "reliability",
            "docker", "kubernetes", "aws", "azure"
        ]

        content_lower = content.lower()

        for keyword in technical_keywords:
            if keyword in content_lower:
                tags.add(keyword)

        return tags

    async def _deduplicate_entities(
        self,
        entities: List[KnowledgeEntity]
    ) -> List[KnowledgeEntity]:
        """Remove duplicate entities"""
        seen = set()
        deduplicated = []

        for entity in entities:
            # Create a signature based on type and content
            signature = (entity.entity_type, entity.content.lower()[:100])

            if signature not in seen:
                seen.add(signature)
                deduplicated.append(entity)

        return deduplicated

    async def _link_related_entities(
        self,
        entities: List[KnowledgeEntity]
    ):
        """Link related entities based on content similarity"""
        # Simple linking based on shared tags
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j:
                    continue

                # Check for shared tags
                shared_tags = entity1.tags & entity2.tags

                if shared_tags and len(shared_tags) >= 2:
                    entity1.related_entities.append(entity2.content[:50])
                    entity2.related_entities.append(entity1.content[:50])

    def _generate_summary(
        self,
        entities: List[KnowledgeEntity]
    ) -> Dict[str, int]:
        """Generate extraction summary"""
        summary = {entity_type.value: 0 for entity_type in KnowledgeEntityType}

        for entity in entities:
            summary[entity.entity_type.value] += 1

        return summary

    async def extract_from_multiple_artifacts(
        self,
        artifacts: List[Dict[str, str]]
    ) -> List[ExtractionResult]:
        """
        Extract knowledge from multiple artifacts.

        Args:
            artifacts: List of {"artifact_id": str, "content": str, "artifact_type": str}

        Returns:
            List of extraction results
        """
        results = []

        for artifact in artifacts:
            result = await self.extract_from_artifact(
                artifact_id=artifact["artifact_id"],
                content=artifact["content"],
                artifact_type=artifact.get("artifact_type", "solution")
            )
            results.append(result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            **self.extraction_stats,
            "average_entities_per_artifact": (
                self.extraction_stats["entities_extracted"] /
                self.extraction_stats["artifacts_processed"]
                if self.extraction_stats["artifacts_processed"] > 0 else 0
            )
        }

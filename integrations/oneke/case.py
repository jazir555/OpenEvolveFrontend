"""
OneKE Case Data Structures

This module defines the data structures for storing and managing
extraction cases in the OneKE case repository.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import hashlib


@dataclass
class Case:
    """
    Represents a single extraction case for case-based learning.

    Attributes:
        case_id: Unique identifier for the case
        input_text: Original input text
        extracted_data: Extracted knowledge (entities, relations, etc.)
        schema: Schema name or definition used for extraction
        domain: Domain label (e.g., 'physics', 'chemistry')
        quality_score: Human-annotated or computed quality (0-1)
        metadata: Additional metadata (timestamp, extractor version, etc.)
        created_at: Case creation timestamp
        updated_at: Last update timestamp
    """
    case_id: str
    input_text: str
    extracted_data: Dict[str, Any]
    schema: str
    domain: str
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert case to dictionary for serialization."""
        return {
            'case_id': self.case_id,
            'input_text': self.input_text,
            'extracted_data': self.extracted_data,
            'schema': self.schema,
            'domain': self.domain,
            'quality_score': self.quality_score,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Case':
        """Create case from dictionary."""
        # Handle datetime parsing
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        return cls(**data)

    @classmethod
    def create(
        cls,
        input_text: str,
        extracted_data: Dict[str, Any],
        schema: str,
        domain: str,
        quality_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Case':
        """
        Create a new case with auto-generated ID.

        Args:
            input_text: Original input text
            extracted_data: Extracted knowledge
            schema: Schema name or definition
            domain: Domain label
            quality_score: Quality score (0-1)
            metadata: Optional metadata

        Returns:
            New Case instance
        """
        # Generate unique ID from content hash
        content_hash = hashlib.sha256(
            f"{input_text}{schema}{domain}".encode()
        ).hexdigest()[:16]

        case_id = f"{domain}_{content_hash}"

        return cls(
            case_id=case_id,
            input_text=input_text,
            extracted_data=extracted_data,
            schema=schema,
            domain=domain,
            quality_score=quality_score,
            metadata=metadata or {}
        )

    def update_quality(self, new_quality: float) -> None:
        """Update quality score and timestamp."""
        self.quality_score = new_quality
        self.updated_at = datetime.utcnow()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata key-value pair."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()


@dataclass
class CaseSimilarity:
    """
    Similarity score between a query and a case.

    Attributes:
        case: The similar case
        similarity: Cosine similarity score (0-1)
        match_reasons: List of reasons why they match
    """
    case: Case
    similarity: float
    match_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'case_id': self.case.case_id,
            'similarity': self.similarity,
            'match_reasons': self.match_reasons,
            'case': self.case.to_dict()
        }


@dataclass
class QualityScore:
    """
    Quality score for an extraction result.

    Attributes:
        completeness: Fraction of required entities present (0-1)
        accuracy: Fraction of entities matching schema (0-1)
        consistency: Absence of contradictions (0-1)
        confidence: Average entity confidence (0-1)
        overall: Overall quality score (weighted average)
    """
    completeness: float
    accuracy: float
    consistency: float
    confidence: float
    overall: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'consistency': self.consistency,
            'confidence': self.confidence,
            'overall': self.overall
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityScore':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ReflectionResult:
    """
    Result of reflection-based improvement.

    Attributes:
        refined_extraction: Improved extraction
        original_quality: Quality before reflection
        refined_quality: Quality after reflection
        issues_found: List of identified issues
        improvements_made: List of improvements applied
        iterations: Number of reflection iterations
    """
    refined_extraction: Dict[str, Any]
    original_quality: QualityScore
    refined_quality: QualityScore
    issues_found: List[str]
    improvements_made: List[str]
    iterations: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'refined_extraction': self.refined_extraction,
            'original_quality': self.original_quality.to_dict(),
            'refined_quality': self.refined_quality.to_dict(),
            'issues_found': self.issues_found,
            'improvements_made': self.improvements_made,
            'iterations': self.iterations
        }


@dataclass
class ConsistencyResult:
    """
    Result of self-consistency checking.

    Attributes:
        is_consistent: Whether samples are consistent
        agreement_ratio: Fraction of samples in agreement (0-1)
        samples: Multiple extraction samples
        consensus_extraction: Consensus extraction
        disagreements: List of disagreement points
    """
    is_consistent: bool
    agreement_ratio: float
    samples: List[Dict[str, Any]]
    consensus_extraction: Dict[str, Any]
    disagreements: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_consistent': self.is_consistent,
            'agreement_ratio': self.agreement_ratio,
            'num_samples': len(self.samples),
            'consensus_extraction': self.consensus_extraction,
            'disagreements': self.disagreements
        }


@dataclass
class EnhancedResult:
    """
    Result of enhanced extraction with quality improvement.

    Attributes:
        extraction: Final enhanced extraction
        quality_score: Final quality score
        original_quality: Quality before enhancement
        quality_improvement: Improvement fraction (0-1)
        strategies_applied: List of enhancement strategies used
        reflection_result: Optional reflection result
        consistency_result: Optional consistency result
        metadata: Additional metadata
    """
    extraction: Dict[str, Any]
    quality_score: QualityScore
    original_quality: QualityScore
    quality_improvement: float
    strategies_applied: List[str]
    reflection_result: Optional[ReflectionResult] = None
    consistency_result: Optional[ConsistencyResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'extraction': self.extraction,
            'quality_score': self.quality_score.to_dict(),
            'original_quality': self.original_quality.to_dict(),
            'quality_improvement': self.quality_improvement,
            'strategies_applied': self.strategies_applied,
            'reflection_result': self.reflection_result.to_dict() if self.reflection_result else None,
            'consistency_result': self.consistency_result.to_dict() if self.consistency_result else None,
            'metadata': self.metadata
        }


@dataclass
class CaseStatistics:
    """
    Statistics about the case repository.

    Attributes:
        total_cases: Total number of cases
        average_quality: Average quality score
        domain_distribution: Cases per domain
        quality_distribution: Quality score distribution (histogram)
        recent_cases: Most recently added cases
    """
    total_cases: int
    average_quality: float
    domain_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]  # e.g., {'0.0-0.2': 5, '0.2-0.4': 10, ...}
    recent_cases: List[str]  # Case IDs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_cases': self.total_cases,
            'average_quality': self.average_quality,
            'domain_distribution': self.domain_distribution,
            'quality_distribution': self.quality_distribution,
            'recent_cases': self.recent_cases
        }

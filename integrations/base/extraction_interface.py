"""
Base Extraction Interface for OpenEvolve

This module defines the abstract interface that all information extraction
implementations must follow. It provides a consistent API for schema-guided
information extraction across different backends (OneKE, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass


class ExtractionType(Enum):
    """Types of information extraction supported."""
    NER = "ner"  # Named Entity Recognition
    RE = "re"   # Relation Extraction
    EE = "ee"   # Event Extraction
    TRIPLE = "triple"  # Triple Extraction
    SCHEMA = "schema"  # Schema-guided extraction


@dataclass
class ExtractionResult:
    """
    Result of an information extraction operation.

    Attributes:
        extraction_type: Type of extraction performed
        entities: List of extracted entities (for NER)
        relations: List of extracted relations (for RE)
        events: List of extracted events (for EE)
        triples: List of extracted triples (for Triple extraction)
        schema: Schema used for extraction
        confidence: Overall confidence score (0-1)
        metadata: Additional metadata about the extraction
        raw_response: Raw response from the extraction backend
    """
    extraction_type: ExtractionType
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    triples: List[Dict[str, Any]]
    schema: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class SchemaDefinition:
    """
    Schema definition for guided information extraction.

    Attributes:
        name: Unique name for the schema
        description: Human-readable description
        entity_types: List of entity types to extract
        relation_types: List of relation types to extract
        event_types: List of event types to extract
        constraints: Optional constraints on extraction
        examples: Example extractions for few-shot learning
    """
    name: str
    description: str
    entity_types: List[Dict[str, Any]]
    relation_types: Optional[List[Dict[str, Any]]] = None
    event_types: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[Dict[str, Any]] = None
    examples: Optional[List[Dict[str, Any]]] = None


class ExtractionInterface(ABC):
    """
    Abstract base class for information extraction implementations.

    This interface defines the contract that all extraction adapters must implement,
    ensuring consistency across different extraction technologies (OneKE, etc.).
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the extraction system with the given configuration.

        Args:
            config: Configuration dictionary containing:
                - model_category: Model type (e.g., "ChatGPT", "LLaMA")
                - model_name_or_path: Model identifier
                - api_key: API key (if applicable)
                - docker: Whether to use Docker (default: false)
                - max_workers: Maximum parallel workers
                - timeout: Request timeout in seconds

        Returns:
            True if initialization was successful, False otherwise.

        Raises:
            ConfigurationError: If configuration is invalid
            ConnectionError: If connection to backend fails
        """
        pass

    @abstractmethod
    async def extract_ner(
        self,
        text: str,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform Named Entity Recognition (NER) extraction.

        Args:
            text: Input text to extract entities from
            schema: Optional schema definition to guide extraction
            **kwargs: Additional parameters for the extraction

        Returns:
            ExtractionResult containing extracted entities

        Raises:
            ExtractionError: If extraction fails
        """
        pass

    @abstractmethod
    async def extract_re(
        self,
        text: str,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform Relation Extraction (RE).

        Args:
            text: Input text to extract relations from
            schema: Optional schema definition to guide extraction
            **kwargs: Additional parameters for the extraction

        Returns:
            ExtractionResult containing extracted relations

        Raises:
            ExtractionError: If extraction fails
        """
        pass

    @abstractmethod
    async def extract_ee(
        self,
        text: str,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform Event Extraction (EE).

        Args:
            text: Input text to extract events from
            schema: Optional schema definition to guide extraction
            **kwargs: Additional parameters for the extraction

        Returns:
            ExtractionResult containing extracted events

        Raises:
            ExtractionError: If extraction fails
        """
        pass

    @abstractmethod
    async def extract_triple(
        self,
        text: str,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform Triple Extraction (subject-relation-object).

        Args:
            text: Input text to extract triples from
            schema: Optional schema definition to guide extraction
            **kwargs: Additional parameters for the extraction

        Returns:
            ExtractionResult containing extracted triples

        Raises:
            ExtractionError: If extraction fails
        """
        pass

    @abstractmethod
    async def extract_schema_guided(
        self,
        text: str,
        schema: SchemaDefinition,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform schema-guided information extraction.

        This is the most flexible extraction method, allowing custom schemas
        that define entities, relations, and events to extract.

        Args:
            text: Input text to extract from
            schema: Schema definition guiding the extraction
            **kwargs: Additional parameters for the extraction

        Returns:
            ExtractionResult containing all extracted information

        Raises:
            ExtractionError: If extraction fails
        """
        pass

    @abstractmethod
    async def batch_extract(
        self,
        texts: List[str],
        extraction_type: ExtractionType,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> List[ExtractionResult]:
        """
        Perform batch extraction on multiple texts.

        Args:
            texts: List of input texts
            extraction_type: Type of extraction to perform
            schema: Optional schema definition
            **kwargs: Additional parameters

        Returns:
            List of ExtractionResult objects

        Raises:
            ExtractionError: If batch extraction fails
        """
        pass

    @abstractmethod
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the extraction system configuration and connection.

        Returns:
            Dictionary containing validation results:
            - is_valid: Overall validation status
            - checks: Individual check results
            - issues: List of any issues found
            - performance: Performance metrics

        Raises:
            ValidationError: If validation itself fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the extraction system.

        Performs cleanup and closes connections to the backend.

        Returns:
            True if shutdown was successful, False otherwise

        Raises:
            ShutdownError: If shutdown fails
        """
        pass

    @abstractmethod
    def load_schema(self, schema_path: str) -> SchemaDefinition:
        """
        Load a schema definition from a file.

        Args:
            schema_path: Path to schema YAML file

        Returns:
            SchemaDefinition object

        Raises:
            SchemaLoadError: If schema loading fails
        """
        pass

    @abstractmethod
    async def extract_from_workflow(
        self,
        workflow_data: Dict[str, Any],
        schemas: List[SchemaDefinition],
        **kwargs
    ) -> Dict[str, ExtractionResult]:
        """
        Extract knowledge from workflow execution data.

        This is a specialized method for extracting structured knowledge
        from OpenEvolve workflow executions.

        Args:
            workflow_data: Workflow execution data
            schemas: List of schemas to apply
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping schema names to ExtractionResult objects

        Raises:
            ExtractionError: If extraction fails
        """
        pass


# Custom Exceptions

class ExtractionError(Exception):
    """Base exception for extraction operations."""
    pass


class ConfigurationError(ExtractionError):
    """Raised when configuration is invalid."""
    pass


class ConnectionError(ExtractionError):
    """Raised when connection to backend fails."""
    pass


class ValidationError(ExtractionError):
    """Raised when validation fails."""
    pass


class SchemaLoadError(ExtractionError):
    """Raised when schema loading fails."""
    pass


class ShutdownError(ExtractionError):
    """Raised when shutdown fails."""
    pass

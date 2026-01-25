"""
Base Domain Knowledge Interface

Abstract interface for domain-specific knowledge integrations into OpenEvolve.
This interface defines the contract that all domain knowledge adapters must implement.

Author: global-chem Integration Specialist
Created: 2026-01-02
Status: ✅ Complete
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class KnowledgeDomain(Enum):
    """Enumeration of supported knowledge domains"""

    CHEMICAL = "chemical"
    BIOLOGICAL = "biological"
    PHYSICAL = "physical"
    MATHEMATICAL = "mathematical"
    MEDICAL = "medical"
    MATERIALS = "materials"
    ENVIRONMENTAL = "environmental"


@dataclass
class KnowledgeArtifact:
    """
    Standardized knowledge artifact representation.

    This is the universal format for knowledge items across all domain integrations.
    """

    # Core identification
    name: str
    domain: KnowledgeDomain
    artifact_type: str

    # Chemical/Biological-specific fields
    smiles: Optional[str] = None
    smarts: Optional[str] = None
    inchi: Optional[str] = None
    inchi_key: Optional[str] = None

    # Properties
    properties: Optional[Dict[str, Any]] = None

    # Metadata
    source: Optional[str] = None
    confidence: Optional[float] = None
    references: Optional[List[str]] = None

    # Additional domain-specific data
    extra_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values for optional fields"""
        if self.properties is None:
            self.properties = {}
        if self.extra_data is None:
            self.extra_data = {}
        if self.references is None:
            self.references = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "domain": self.domain.value,
            "artifact_type": self.artifact_type,
            "smiles": self.smiles,
            "smarts": self.smarts,
            "inchi": self.inchi,
            "inchi_key": self.inchi_key,
            "properties": self.properties,
            "source": self.source,
            "confidence": self.confidence,
            "references": self.references,
            "extra_data": self.extra_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeArtifact":
        """Create from dictionary representation"""
        return cls(
            name=data["name"],
            domain=KnowledgeDomain(data["domain"]),
            artifact_type=data["artifact_type"],
            smiles=data.get("smiles"),
            smarts=data.get("smarts"),
            inchi=data.get("inchi"),
            inchi_key=data.get("inchi_key"),
            properties=data.get("properties"),
            source=data.get("source"),
            confidence=data.get("confidence"),
            references=data.get("references"),
            extra_data=data.get("extra_data"),
        )


@dataclass
class QueryResult:
    """Result from a domain knowledge query"""

    artifacts: List[KnowledgeArtifact]
    total_found: int
    query_time_ms: float
    source: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DomainKnowledgeInterface(ABC):
    """
    Abstract base class for domain knowledge integrations.

    All domain-specific adapters must inherit from this class and implement
    the required methods. This ensures a consistent interface across all
    knowledge domains.

    Critical Requirements:
    - Must be async-compatible
    - Must implement graceful degradation
    - Must provide validation capabilities
    - Must support configuration
    - Must handle errors gracefully
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the domain knowledge adapter.

        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config or {}
        self._initialized = False
        self._available = False

    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the domain knowledge adapter.

        This method should:
        - Load any necessary data/models
        - Establish connections to external services
        - Validate availability
        - Set up internal state

        Args:
            config: Optional configuration override

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def validate(self) -> bool:
        """
        Validate that the adapter is functioning correctly.

        This should check:
        - Data integrity
        - Connection status
        - Basic query functionality

        Returns:
            True if validation passes, False otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Clean shutdown of the adapter.

        This should:
        - Close connections
        - Release resources
        - Persist any necessary state
        """
        pass

    # Chemical-specific methods (for chemical domain)

    @abstractmethod
    async def query_chemical(
        self,
        name: str,
        fuzzy_match: bool = True,
        max_results: int = 10,
    ) -> QueryResult:
        """
        Query for chemical information by name.

        Args:
            name: Chemical name or identifier
            fuzzy_match: Whether to use fuzzy matching
            max_results: Maximum number of results to return

        Returns:
            QueryResult containing matching chemical artifacts
        """
        pass

    @abstractmethod
    async def search_smiles(
        self,
        smiles: str,
        exact_match: bool = False,
        max_results: int = 10,
    ) -> QueryResult:
        """
        Search for chemicals by SMILES string.

        Args:
            smiles: SMILES string to search for
            exact_match: Whether to require exact match
            max_results: Maximum number of results

        Returns:
            QueryResult with matching chemicals
        """
        pass

    @abstractmethod
    async def get_properties(
        self,
        name: str,
        property_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get properties for a specific chemical.

        Args:
            name: Chemical name or identifier
            property_names: Specific properties to retrieve (None = all)

        Returns:
            Dictionary of property names to values
        """
        pass

    # General domain knowledge methods

    @abstractmethod
    async def search(
        self,
        query: str,
        domain: Optional[KnowledgeDomain] = None,
        max_results: int = 10,
    ) -> QueryResult:
        """
        General search across the knowledge base.

        Args:
            query: Search query string
            domain: Optional domain filter
            max_results: Maximum results to return

        Returns:
            QueryResult with matching artifacts
        """
        pass

    @abstractmethod
    async def get_available_categories(self) -> List[str]:
        """
        Get list of available knowledge categories.

        Returns:
            List of category names
        """
        pass

    # Utility methods

    @property
    def is_initialized(self) -> bool:
        """Check if adapter has been initialized"""
        return self._initialized

    @property
    def is_available(self) -> bool:
        """Check if adapter is available for use"""
        return self._available

    def get_domain(self) -> KnowledgeDomain:
        """Get the primary domain of this adapter"""
        raise NotImplementedError("Subclasses must implement get_domain()")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check.

        Returns:
            Dictionary with health status information
        """
        health = {
            "initialized": self.is_initialized,
            "available": self.is_available,
            "domain": self.get_domain().value,
            "config": self.config,
        }

        if self.is_initialized:
            health["valid"] = await self.validate()

        return health


class DomainKnowledgeException(Exception):
    """Base exception for domain knowledge operations"""

    def __init__(self, message: str, domain: str, details: Optional[Dict] = None):
        self.message = message
        self.domain = domain
        self.details = details or {}
        super().__init__(f"[{domain}] {message}")


class QueryException(DomainKnowledgeException):
    """Exception raised during query operations"""

    pass


class InitializationException(DomainKnowledgeException):
    """Exception raised during initialization"""

    pass


class ValidationException(DomainKnowledgeException):
    """Exception raised during validation"""

    pass

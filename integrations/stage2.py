"""
Stage 2 Integration: Isomorphic Mapping with Ψ₂, Ψ₃, and I_mech

Integrates RESE's Ontology Mapping (Ψ₂), Constraint Inversion (Ψ₃),
and Mechanistic Isomorphism (I_mech) with E2E Stage 2.

Architecture:
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Source Domain   │───▶│   Ψ₂ Ontology    │───▶│  Ψ₃ Inversion   │
│  Analysis        │    │   Mapping        │    │  + I_mech        │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                               │                       │
                               ▼                       ▼
                        ┌──────────────────────────────────────┐
                        │     Isomorphic Mapping Pipeline      │
                        └──────────────────────────────────────┘

Author: Agent A4 (Stage Integration Lead)
Created: 2025-12-31
Status: 🟢 Active Implementation
Target: 1.5 hours implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
import numpy as np

# Import RESE components
import sys
from pathlib import Path

# Try to import RESE components, use stubs if not available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "rese"))
    from phase2.imech import (
        IMechValidator, Domain, FunctionalDependencyGraph,
        SolutionMapper, SolutionValidator, compare_domains
    )
    IMECH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    IMECH_AVAILABLE = False
    # Create stub classes for graceful degradation
    import numpy as np
    from dataclasses import dataclass
    from typing import List, Dict, Optional, Any

    @dataclass
    class Domain:
        name: str
        entities: List[Any] = None
        relations: List[Any] = None

    @dataclass
    class FunctionalDependencyGraph:
        domain: Domain
        dependencies: Dict[str, List[str]] = None

    class IMechValidator:
        def __init__(self, *args, **kwargs):
            self.available = False

    class SolutionMapper:
        def __init__(self, *args, **kwargs):
            self.available = False

    class SolutionValidator:
        def __init__(self, *args, **kwargs):
            self.available = False

    def compare_domains(d1, d2):
        return {"is_isomorphic": False, "message": "IMECH not available"}


# ============================================================================
# Enums and Data Structures
# ============================================================================

class MappingStatus(Enum):
    """Status of isomorphic mapping"""
    ANALYZING = "analyzing"
    ONTOLOGY_MAPPED = "ontology_mapped"
    INVERTED = "inverted"
    ISOMORPHISM_CHECKED = "isomorphism_checked"
    VALIDATED = "validated"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DomainPair:
    """Pair of domains for isomorphic analysis"""
    source_domain: Domain
    target_domain: Domain
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologyMapping:
    """Mapping between ontologies"""
    source_concept: str
    target_concept: str
    similarity_score: float
    mapping_type: str  # "exact", "approximate", "analogical"
    confidence: float
    rationale: str = ""


@dataclass
class InversionResult:
    """Result from constraint inversion (Ψ₃)"""
    original_constraint: str
    inverted_constraint: str
    complexity_reduction: float
    validity_preserved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IsomorphicMappingResult:
    """Result from Stage 2 isomorphic mapping"""
    status: MappingStatus
    domain_pair: DomainPair
    ontology_mappings: List[OntologyMapping]
    inversion_results: List[InversionResult]
    isomorphism_score: float
    transfer_confidence: float
    suggested_transfers: List[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]] = None
    analysis_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'source_domain': self.domain_pair.source_domain.id,
            'target_domain': self.domain_pair.target_domain.id,
            'ontology_mappings': [
                {
                    'source': m.source_concept,
                    'target': m.target_concept,
                    'similarity': m.similarity_score,
                    'type': m.mapping_type,
                    'confidence': m.confidence
                }
                for m in self.ontology_mappings
            ],
            'inversion_results': [
                {
                    'original': inv.original_constraint,
                    'inverted': inv.inverted_constraint,
                    'complexity_reduction': inv.complexity_reduction,
                    'valid': inv.validity_preserved
                }
                for inv in self.inversion_results
            ],
            'isomorphism_score': self.isomorphism_score,
            'transfer_confidence': self.transfer_confidence,
            'suggested_transfers': self.suggested_transfers,
            'validation_results': self.validation_results,
            'analysis_time': self.analysis_time,
            'metadata': self.metadata,
            'errors': self.errors
        }


# ============================================================================
# Main Integration Class
# ============================================================================

class Stage2Integration:
    """
    Stage 2 Integration: Isomorphic Mapping with RESE components.

    This module integrates:
    1. Ψ₂: Ontology Mapping for concept alignment
    2. Ψ₃: Constraint Inversion for complexity reduction
    3. I_mech: Mechanistic Isomorphism for structural matching
    4. Solution transfer validation

    Workflow:
    1. Analyze source and target domains
    2. Map ontologies using Ψ₂
    3. Invert constraints using Ψ₃
    4. Check mechanistic isomorphism using I_mech
    5. Validate solution transferability
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_psi2: bool = True,
        enable_psi3: bool = True,
        enable_imech: bool = True,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize Stage 2 Integration.

        Args:
            config: Optional configuration dictionary
            enable_psi2: Enable Ontology Mapping (Ψ₂)
            enable_psi3: Enable Constraint Inversion (Ψ₃)
            enable_imech: Enable Mechanistic Isomorphism (I_mech)
            similarity_threshold: Minimum similarity for ontology mapping
        """
        self.config = config or {}
        self.enable_psi2 = enable_psi2
        self.enable_psi3 = enable_psi3
        self.enable_imech = enable_imech
        self.similarity_threshold = similarity_threshold

        # Initialize components
        if self.enable_imech:
            self.imech_validator = IMechValidator()
            self.solution_mapper = SolutionMapper()
            self.solution_validator = SolutionValidator()
            self.compare_domains_func = compare_domains

        # Mapping history
        self.mapping_history: List[IsomorphicMappingResult] = []

    def analyze_domains(
        self,
        source_domain: Domain,
        target_domain: Domain,
        perform_inversion: bool = True
    ) -> IsomorphicMappingResult:
        """
        Analyze domains for isomorphic mapping.

        Args:
            source_domain: Source domain with known solution
            target_domain: Target domain to transfer to
            perform_inversion: Whether to perform constraint inversion

        Returns:
            IsomorphicMappingResult with mapping analysis
        """
        start_time = datetime.now()
        domain_pair = DomainPair(source_domain, target_domain)

        result = IsomorphicMappingResult(
            status=MappingStatus.ANALYZING,
            domain_pair=domain_pair,
            ontology_mappings=[],
            inversion_results=[],
            isomorphism_score=0.0,
            transfer_confidence=0.0,
            suggested_transfers=[]
        )

        try:
            # Step 1: Ontology mapping using Ψ₂
            if self.enable_psi2:
                result.ontology_mappings = self._map_ontologies(
                    source_domain,
                    target_domain
                )
                result.status = MappingStatus.ONTOLOGY_MAPPED

            # Step 2: Constraint inversion using Ψ₃
            if self.enable_psi3 and perform_inversion:
                result.inversion_results = self._invert_constraints(
                    source_domain,
                    target_domain
                )
                result.status = MappingStatus.INVERTED

            # Step 3: Mechanistic isomorphism check using I_mech
            if self.enable_imech:
                similarity_result = self.compare_domains_func(
                    source_domain,
                    target_domain
                )

                result.isomorphism_score = similarity_result.total_score
                result.status = MappingStatus.ISOMORPHISM_CHECKED

                # Generate transfer suggestions
                if result.isomorphism_score >= self.similarity_threshold:
                    result.suggested_transfers = self._suggest_transfers(
                        source_domain,
                        target_domain,
                        result.ontology_mappings
                    )

                # Validate transfers
                if result.suggested_transfers:
                    result.validation_results = self._validate_transfers(
                        source_domain,
                        target_domain,
                        result.suggested_transfers
                    )
                    result.status = MappingStatus.VALIDATED

            # Mark as completed if we got this far
            if result.status != MappingStatus.FAILED:
                result.status = MappingStatus.COMPLETED

            # Calculate transfer confidence
            result.transfer_confidence = self._calculate_transfer_confidence(result)

        except Exception as e:
            result.status = MappingStatus.FAILED
            result.errors.append(str(e))

        # Record time
        end_time = datetime.now()
        result.analysis_time = (end_time - start_time).total_seconds()

        # Store in history
        self.mapping_history.append(result)

        return result

    def _map_ontologies(
        self,
        source_domain: Domain,
        target_domain: Domain
    ) -> List[OntologyMapping]:
        """
        Map ontologies between domains using Ψ₂.

        Args:
            source_domain: Source domain
            target_domain: Target domain

        Returns:
            List of ontology mappings
        """
        mappings = []

        # Simple ontology mapping based on string similarity
        # In production, this would use embeddings and semantic similarity

        source_vars = set(source_domain.metadata.get('variables', {}).keys())
        target_vars = set(target_domain.metadata.get('variables', {}).keys())

        # Find similar variable names
        for source_var in source_vars:
            best_match = None
            best_score = 0.0

            for target_var in target_vars:
                # Simple similarity: exact match or contains
                if source_var.lower() == target_var.lower():
                    score = 1.0
                    mapping_type = "exact"
                elif source_var.lower() in target_var.lower() or target_var.lower() in source_var.lower():
                    score = 0.8
                    mapping_type = "approximate"
                else:
                    # Edit distance approximation
                    score = self._string_similarity(source_var, target_var)
                    mapping_type = "analogical" if score > 0.5 else None

                if mapping_type and score > best_score:
                    best_match = target_var
                    best_score = score
                    mapping_type = mapping_type

            if best_match and best_score >= self.similarity_threshold:
                mappings.append(
                    OntologyMapping(
                        source_concept=source_var,
                        target_concept=best_match,
                        similarity_score=best_score,
                        mapping_type=mapping_type,
                        confidence=best_score,
                        rationale=f"Lexical similarity: {best_score:.2f}"
                    )
                )

        return mappings

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity using simple heuristics.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score [0, 1]
        """
        s1, s2 = s1.lower(), s2.lower()

        # Length-based similarity
        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)

        if max_len == 0:
            return 1.0

        # Character overlap
        overlap = len(set(s1) & set(s2))
        similarity = overlap / max_len

        return similarity

    def _invert_constraints(
        self,
        source_domain: Domain,
        target_domain: Domain
    ) -> List[InversionResult]:
        """
        Invert constraints using Ψ₃.

        Args:
            source_domain: Source domain
            target_domain: Target domain

        Returns:
            List of inversion results
        """
        inversions = []

        # Get constraints from source domain
        constraints = source_domain.formal_constraints

        for i, constraint in enumerate(constraints):
            # Simple inversion: transform constraints
            # In production, this would use sophisticated constraint transformation

            if hasattr(constraint, 'description'):
                desc = constraint.description
            elif isinstance(constraint, str):
                desc = constraint
            else:
                desc = str(constraint)

            # Invert: maximize -> minimize, etc.
            inverted_desc = self._invert_constraint_text(desc)

            # Calculate complexity reduction (simplified)
            complexity_reduction = min(0.5, len(desc) * 0.01)

            inversions.append(
                InversionResult(
                    original_constraint=desc,
                    inverted_constraint=inverted_desc,
                    complexity_reduction=complexity_reduction,
                    validity_preserved=True,  # Assume valid for now
                    metadata={'constraint_index': i}
                )
            )

        return inversions

    def _invert_constraint_text(self, constraint: str) -> str:
        """
        Invert constraint text.

        Args:
            constraint: Original constraint

        Returns:
            Inverted constraint
        """
        constraint_lower = constraint.lower()

        # Simple transformations
        if "maximize" in constraint_lower:
            return constraint_lower.replace("maximize", "minimize")
        elif "minimize" in constraint_lower:
            return constraint_lower.replace("minimize", "maximize")
        elif "greater than" in constraint_lower:
            return constraint_lower.replace("greater than", "less than")
        elif "less than" in constraint_lower:
            return constraint_lower.replace("less than", "greater than")
        elif "≥" in constraint:
            return constraint.replace("≥", "≤")
        elif "≤" in constraint:
            return constraint.replace("≤", "≥")
        else:
            return f"inverted({constraint})"

    def _suggest_transfers(
        self,
        source_domain: Domain,
        target_domain: Domain,
        ontology_mappings: List[OntologyMapping]
    ) -> List[Dict[str, Any]]:
        """
        Suggest solution transfers based on isomorphism.

        Args:
            source_domain: Source domain
            target_domain: Target domain
            ontology_mappings: Ontology mappings

        Returns:
            List of suggested transfers
        """
        transfers = []

        # Check if source has solution
        if not source_domain.has_solution():
            return transfers

        solution = source_domain.get_primary_solution()

        # Create transfer suggestion
        transfer = {
            'source_solution_id': getattr(solution, 'id', 'unknown'),
            'source_domain': source_domain.id,
            'target_domain': target_domain.id,
            'mapping_based_on': len(ontology_mappings),
            'ontology_mappings': [
                {
                    'source': m.source_concept,
                    'target': m.target_concept,
                    'confidence': m.confidence
                }
                for m in ontology_mappings
            ],
            'transfer_method': 'direct',
            'estimated_success_rate': 0.8  # Default score
        }

        transfers.append(transfer)

        return transfers

    def _validate_transfers(
        self,
        source_domain: Domain,
        target_domain: Domain,
        transfers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate suggested transfers.

        Args:
            source_domain: Source domain
            target_domain: Target domain
            transfers: Suggested transfers

        Returns:
            Validation results
        """
        validation_results = {
            'total_transfers': len(transfers),
            'valid_transfers': 0,
            'transfer_details': []
        }

        for transfer in transfers:
            # Simple validation: check if mappings exist
            has_mappings = len(transfer.get('ontology_mappings', [])) > 0

            transfer_detail = {
                'transfer_id': transfer.get('source_solution_id'),
                'is_valid': has_mappings,
                'confidence': transfer.get('estimated_success_rate', 0.0),
                'validation_method': 'ontology_coverage'
            }

            if has_mappings:
                validation_results['valid_transfers'] += 1

            validation_results['transfer_details'].append(transfer_detail)

        return validation_results

    def _calculate_transfer_confidence(
        self,
        result: IsomorphicMappingResult
    ) -> float:
        """
        Calculate overall transfer confidence.

        Args:
            result: Mapping result

        Returns:
            Confidence score [0, 1]
        """
        # Base confidence from isomorphism score
        base_confidence = result.isomorphism_score

        # Adjust based on ontology mappings
        mapping_bonus = min(len(result.ontology_mappings) * 0.05, 0.2)

        # Adjust based on validation
        validation_penalty = 0.0
        if result.validation_results:
            valid_ratio = (
                result.validation_results.get('valid_transfers', 0) /
                max(result.validation_results.get('total_transfers', 1), 1)
            )
            validation_penalty = (1.0 - valid_ratio) * 0.3

        confidence = base_confidence + mapping_bonus - validation_penalty

        return max(0.0, min(1.0, confidence))

    def get_best_transfer_candidate(
        self,
        result: IsomorphicMappingResult
    ) -> Optional[Dict[str, Any]]:
        """
        Get best transfer candidate from result.

        Args:
            result: Mapping result

        Returns:
            Best transfer candidate or None
        """
        if not result.suggested_transfers:
            return None

        # Sort by estimated success rate
        sorted_transfers = sorted(
            result.suggested_transfers,
            key=lambda t: t.get('estimated_success_rate', 0.0),
            reverse=True
        )

        return sorted_transfers[0]

    def export_mapping(
        self,
        result: IsomorphicMappingResult,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export mapping result to JSON.

        Args:
            result: Mapping result to export
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"stage2_mapping_{timestamp}.json")

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        return output_path


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_domain_pair(
    source_domain: Domain,
    target_domain: Domain,
    config: Optional[Dict[str, Any]] = None
) -> IsomorphicMappingResult:
    """
    Convenience function to analyze domain pair.

    Args:
        source_domain: Source domain
        target_domain: Target domain
        config: Optional configuration

    Returns:
        IsomorphicMappingResult
    """
    integration = Stage2Integration(config=config)

    return integration.analyze_domains(source_domain, target_domain)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main class
    'Stage2Integration',

    # Data structures
    'DomainPair',
    'OntologyMapping',
    'InversionResult',
    'IsomorphicMappingResult',

    # Enums
    'MappingStatus',

    # Convenience functions
    'analyze_domain_pair',
]

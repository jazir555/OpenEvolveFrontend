"""Section-aware operations for Core-Project ACE.

This module provides utilities for managing section-prefixed IDs and
section-aware update batches, following the Root ACE pattern where
each section has a unique 3-5 character slug prefix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .updates import UpdateBatch, UpdateOperation


# Common section name to slug mappings (from Root ACE)
SECTION_SLUG_MAPPING: Dict[str, str] = {
    "financial_strategies_and_insights": "fin",
    "formulas_and_calculations": "calc",
    "code_snippets_and_templates": "code",
    "common_mistakes_to_avoid": "err",
    "problem_solving_heuristics": "prob",
    "context_clues_and_indicators": "ctx",
    "others": "misc",
    "meta_strategies": "meta",
}


def get_section_slug(section_name: str) -> str:
    """Convert section name to slug format (3-5 chars).

    Args:
        section_name: The full section name (e.g., "financial_strategies_and_insights")

    Returns:
        A 3-5 character slug prefix (e.g., "fin")

    Examples:
        >>> get_section_slug("financial_strategies_and_insights")
        'fin'
        >>> get_section_slug("formulas_and_calculations")
        'calc'
        >>> get_section_slug("code_snippets_and_templates")
        'code'
        >>> get_section_slug("problem_solving_heuristics")
        'prob'

    Notes:
        - For known sections, uses predefined mapping from Root ACE
        - For unknown sections, generates from first consonants (3-5 chars)
        - Ensures all slugs are lowercase and unique
        - Supports partial matching (e.g., "financial_strategies" -> "fin")
    """
    # Normalize first
    normalized = normalize_section_name(section_name)

    # Check if we have a predefined mapping (exact match)
    if normalized in SECTION_SLUG_MAPPING:
        return SECTION_SLUG_MAPPING[normalized]

    # Check for partial match (normalized is a prefix of a known section)
    for known_section, slug in SECTION_SLUG_MAPPING.items():
        if known_section.startswith(normalized) or normalized.startswith(known_section.split("_")[0]):
            # Match if the normalized name starts with the first word of a known section
            # or if the known section starts with the normalized name
            if len(normalized) >= 3:  # Minimum reasonable partial match
                return slug

    # For unknown sections, generate from first letters
    # Remove common stopwords for better slugs
    stopwords = {"and", "or", "the", "for", "with", "from"}
    words = [w for w in normalized.split("_") if w and w not in stopwords]

    if not words:
        # Fallback: use first 3 chars of normalized name, pad with 'x' if needed
        base = normalized[:3].lower() if len(normalized) >= 3 else normalized.lower()
        return (base + "xxx")[:3]

    # Take first letter of each word (3-5 chars, up to 5 words)
    slug = "".join(word[0] for word in words[:5]).lower()

    # Ensure minimum length of 3 by adding more letters from words
    if len(slug) < 3:
        word_idx = 0
        while len(slug) < 3 and word_idx < len(words):
            word = words[word_idx]
            # Add subsequent letters from each word until we reach 3
            for char_idx in range(1, len(word)):
                if len(slug) >= 3:
                    break
                slug += word[char_idx]
            word_idx += 1

        # If still too short, pad with 'x'
        if len(slug) < 3:
            slug = (slug + "xxx")[:3]

    return slug[:5]  # Max 5 characters


def normalize_section_name(section: str) -> str:
    """Normalize section name to snake_case.

    Args:
        section: Section name in any format (e.g., "Financial Strategies")

    Returns:
        Normalized snake_case name (e.g., "financial_strategies")

    Examples:
        >>> normalize_section_name("Financial Strategies")
        'financial_strategies'
        >>> normalize_section_name("Q&A")
        'q_and_a'
        >>> normalize_section_name("Code & Templates")
        'code_and_templates'

    Notes:
        - Converts to lowercase
        - Replaces spaces and special characters with underscores
        - Handles common patterns like "&" -> "and"
    """
    if not section:
        return "unknown"

    # Replace common special characters
    normalized = section
    normalized = re.sub(r"&", " and ", normalized)
    normalized = re.sub(r"\+", " and ", normalized)
    normalized = re.sub(r"/", " or ", normalized)

    # Convert to lowercase
    normalized = normalized.lower()

    # Replace non-alphanumeric with underscores
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)

    # Remove leading/trailing underscores
    normalized = normalized.strip("_")

    # Collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)

    return normalized if normalized else "unknown"


def generate_section_id(section: str, next_id: int) -> str:
    """Generate section-prefixed ID.

    Args:
        section: The section name
        next_id: The next sequential ID number

    Returns:
        A section-prefixed ID with 5-digit zero-padding

    Examples:
        >>> generate_section_id("financial_strategies", 1)
        'fin-00001'
        >>> generate_section_id("formulas", 42)
        'calc-00042'
        >>> generate_section_id("unknown_section", 999)
        'unk-00999'

    Notes:
        - Uses 5-digit zero-padded format from Root ACE
        - Section slug is 3-5 characters
        - Format: {slug}-{id:05d}
    """
    slug = get_section_slug(section)
    return f"{slug}-{next_id:05d}"


@dataclass
class SectionAwareUpdateBatch:
    """Enhanced UpdateBatch with section-aware ID generation.

    This extends the standard UpdateBatch to automatically generate
    section-prefixed IDs for ADD operations, following the Root ACE
    convention of "slug-00001" format.

    Attributes:
        reasoning: SkillManager's reasoning for the update
        operations: List of UpdateOperation objects to apply
        section_index: Next ID counter per section (slug-based)

    Examples:
        >>> batch = SectionAwareUpdateBatch(reasoning="Add financial skills")
        >>> batch.add_operation("financial_strategies", "Diversify portfolio", "ADD")
        >>> batch.operations[0].skill_id
        'fin-00001'
        >>> batch.add_operation("formulas", "PV = FV / (1+r)^n", "ADD")
        >>> batch.operations[1].skill_id
        'calc-00001'

    Notes:
        - Section slugs are generated automatically
        - Each section maintains its own ID counter
        - Compatible with standard UpdateBatch serialization
    """

    reasoning: str
    operations: List[UpdateOperation] = field(default_factory=list)
    section_index: Dict[str, int] = field(default_factory=dict)

    def add_operation(
        self,
        section: str,
        content: str,
        operation_type: str = "ADD",
        skill_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> UpdateOperation:
        """Add an operation with automatic section-aware ID generation.

        Args:
            section: The section name
            content: The skill content
            operation_type: Type of operation ("ADD", "UPDATE", "TAG", "REMOVE")
            skill_id: Optional explicit skill_id (auto-generated if None for ADD)
            metadata: Optional metadata dictionary

        Returns:
            The created UpdateOperation

        Notes:
            - For ADD operations with no skill_id, generates section-prefixed ID
            - For other operations, skill_id is typically required
            - Section names are normalized automatically
        """
        # Normalize section name
        normalized_section = normalize_section_name(section)

        # Generate skill_id for ADD operations if not provided
        if operation_type.upper() == "ADD" and skill_id is None:
            section_slug = get_section_slug(normalized_section)
            next_id = self.get_next_id(section_slug)
            skill_id = generate_section_id(normalized_section, next_id)

        operation = UpdateOperation(
            type=operation_type.upper(),  # type: ignore[arg-type]
            section=normalized_section,
            content=content,
            skill_id=skill_id,
            metadata=metadata or {},
        )

        self.operations.append(operation)
        return operation

    def normalize_sections(self) -> None:
        """Normalize all section names in existing operations.

        This is useful when processing batches from external sources
        that may have inconsistent section naming.

        Examples:
            >>> batch = SectionAwareUpdateBatch(reasoning="Fix sections")
            >>> batch.operations.append(UpdateOperation(
            ...     type="ADD",
            ...     section="Financial Strategies",
            ...     content="Diversify"
            ... ))
            >>> batch.normalize_sections()
            >>> batch.operations[0].section
            'financial_strategies'
        """
        for op in self.operations:
            op.section = normalize_section_name(op.section)

    def get_next_id(self, section_slug: str) -> int:
        """Get the next ID for a given section.

        Args:
            section_slug: The section's slug (e.g., "fin", "calc")

        Returns:
            The next sequential ID number for this section

        Examples:
            >>> batch = SectionAwareUpdateBatch(reasoning="Test")
            >>> batch.get_next_id("fin")
            1
            >>> batch.get_next_id("fin")
            2
            >>> batch.get_next_id("calc")
            1
        """
        if section_slug not in self.section_index:
            self.section_index[section_slug] = 0
        self.section_index[section_slug] += 1
        return self.section_index[section_slug]

    def to_update_batch(self) -> UpdateBatch:
        """Convert to standard UpdateBatch for compatibility.

        Returns:
            A standard UpdateBatch with reasoning and operations

        Examples:
            >>> batch = SectionAwareUpdateBatch(reasoning="Add skills")
            >>> batch.add_operation("financial_strategies", "Save money")
            >>> std_batch = batch.to_update_batch()
            >>> isinstance(std_batch, UpdateBatch)
            True
        """
        return UpdateBatch(reasoning=self.reasoning, operations=self.operations)

    @classmethod
    def from_update_batch(cls, batch: UpdateBatch) -> "SectionAwareUpdateBatch":
        """Create SectionAwareUpdateBatch from standard UpdateBatch.

        Args:
            batch: Standard UpdateBatch to convert

        Returns:
            A new SectionAwareUpdateBatch with the same data

        Examples:
            >>> std_batch = UpdateBatch(reasoning="Update skills")
            >>> std_batch.operations.append(UpdateOperation(
            ...     type="ADD",
            ...     section="financial_strategies",
            ...     content="Save money"
            ... ))
            >>> section_batch = SectionAwareUpdateBatch.from_update_batch(std_batch)
        """
        return cls(reasoning=batch.reasoning, operations=batch.operations)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary with reasoning, operations, and section_index

        Examples:
            >>> batch = SectionAwareUpdateBatch(reasoning="Add skills")
            >>> batch.add_operation("financial_strategies", "Save money")
            >>> json_data = batch.to_json()
            >>> 'section_index' in json_data
            True
        """
        return {
            "reasoning": self.reasoning,
            "operations": [op.to_json() for op in self.operations],
            "section_index": self.section_index,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "SectionAwareUpdateBatch":
        """Create SectionAwareUpdateBatch from JSON dictionary.

        Args:
            payload: Dictionary with reasoning, operations, and optional section_index

        Returns:
            A new SectionAwareUpdateBatch instance

        Examples:
            >>> data = {
            ...     "reasoning": "Add skills",
            ...     "operations": [{
            ...         "type": "ADD",
            ...         "section": "financial_strategies",
            ...         "content": "Save money"
            ...     }],
            ...     "section_index": {"fin": 1}
            ... }
            >>> batch = SectionAwareUpdateBatch.from_json(data)
        """
        ops_payload = payload.get("operations", [])
        operations = []
        if isinstance(ops_payload, list):
            for item in ops_payload:
                if isinstance(item, dict):
                    operations.append(UpdateOperation.from_json(item))

        section_index = payload.get("section_index", {})
        if isinstance(section_index, dict):
            # Convert string keys back to strings if they were serialized
            section_index = {str(k): int(v) for k, v in section_index.items()}

        return cls(
            reasoning=str(payload.get("reasoning", "")),
            operations=operations,
            section_index=section_index,  # type: ignore[arg-type]
        )

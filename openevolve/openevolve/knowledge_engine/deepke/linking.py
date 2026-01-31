"""
Entity Linking and Disambiguation

Links extracted entities to knowledge base entries and resolves ambiguities.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher

from .extractor import ExtractedEntity, EntityType

logger = logging.getLogger(__name__)


@dataclass
class EntityCandidate:
    """A candidate entity from the knowledge base"""
    entity_id: str
    name: str
    entity_type: EntityType
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_names(self) -> List[str]:
        """Get all names and aliases"""
        return [self.name] + self.aliases


@dataclass
class LinkingResult:
    """Result of entity linking"""
    entity: ExtractedEntity
    candidates: List[Tuple[EntityCandidate, float]]  # candidate, score
    selected_candidate: Optional[EntityCandidate] = None
    confidence: float = 0.0
    linking_method: str = ""
    
    def is_linked(self) -> bool:
        """Check if entity was successfully linked"""
        return self.selected_candidate is not None


class EntityLinker:
    """Links extracted entities to knowledge base"""
    
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self.similarity_threshold = 0.8
        
        # Entity cache for faster lookups
        self._entity_cache: Dict[str, List[EntityCandidate]] = {}
    
    def add_to_cache(self, candidates: List[EntityCandidate]):
        """Add candidates to cache"""
        for candidate in candidates:
            type_key = candidate.entity_type.value
            if type_key not in self._entity_cache:
                self._entity_cache[type_key] = []
            self._entity_cache[type_key].append(candidate)
    
    def find_candidates(
        self,
        entity: ExtractedEntity,
        max_candidates: int = 5
    ) -> List[Tuple[EntityCandidate, float]]:
        """Find candidate entities from knowledge base"""
        candidates = []
        
        # Check cache first
        type_key = entity.entity_type.value
        cached = self._entity_cache.get(type_key, [])
        
        for candidate in cached:
            # Calculate similarity with all names
            best_score = 0.0
            for name in candidate.get_all_names():
                score = self._calculate_similarity(
                    entity.normalized_text.lower(),
                    name.lower()
                )
                best_score = max(best_score, score)
            
            if best_score >= self.similarity_threshold:
                candidates.append((candidate, best_score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate string similarity"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def link(
        self,
        entity: ExtractedEntity,
        context: Optional[str] = None
    ) -> LinkingResult:
        """Link an entity to knowledge base"""
        candidates = self.find_candidates(entity)
        
        if not candidates:
            return LinkingResult(
                entity=entity,
                candidates=[],
                confidence=0.0,
                linking_method="no_match"
            )
        
        # Select best candidate
        best_candidate, best_score = candidates[0]
        
        # Use context to disambiguate if multiple good candidates
        if len(candidates) > 1 and context:
            selected = self._disambiguate_with_context(
                entity, candidates, context
            )
        else:
            selected = best_candidate
        
        return LinkingResult(
            entity=entity,
            candidates=candidates,
            selected_candidate=selected,
            confidence=best_score,
            linking_method="exact_match" if best_score > 0.95 else "similarity"
        )
    
    def _disambiguate_with_context(
        self,
        entity: ExtractedEntity,
        candidates: List[Tuple[EntityCandidate, float]],
        context: str
    ) -> EntityCandidate:
        """Use context to disambiguate between candidates"""
        # Simple context matching - check if candidate description appears in context
        context_lower = context.lower()
        
        best_candidate = candidates[0][0]
        best_context_score = 0.0
        
        for candidate, base_score in candidates:
            context_score = 0.0
            
            # Check description overlap
            if candidate.description.lower() in context_lower:
                context_score += 0.3
            
            # Check alias overlap
            for alias in candidate.aliases:
                if alias.lower() in context_lower:
                    context_score += 0.1
            
            # Combined score
            total_score = base_score + context_score
            if total_score > best_context_score:
                best_context_score = total_score
                best_candidate = candidate
        
        return best_candidate
    
    def link_batch(
        self,
        entities: List[ExtractedEntity],
        context: Optional[str] = None
    ) -> List[LinkingResult]:
        """Link multiple entities"""
        return [self.link(e, context) for e in entities]


class EntityDisambiguator:
    """Disambiguates between multiple entity interpretations"""
    
    def __init__(self):
        self.entity_graph: Dict[str, List[str]] = {}  # entity -> related entities
    
    def build_entity_graph(self, relations: List):
        """Build graph from relations for coherence scoring"""
        for relation in relations:
            subj = relation.subject.normalized_text
            obj = relation.object.normalized_text
            
            if subj not in self.entity_graph:
                self.entity_graph[subj] = []
            if obj not in self.entity_graph:
                self.entity_graph[obj] = []
            
            self.entity_graph[subj].append(obj)
            self.entity_graph[obj].append(subj)
    
    def disambiguate(
        self,
        ambiguous_entities: List[LinkingResult],
        context_entities: List[ExtractedEntity]
    ) -> List[LinkingResult]:
        """Disambiguate entities using coherence scoring"""
        resolved = []
        
        for linking_result in ambiguous_entities:
            if not linking_result.candidates:
                resolved.append(linking_result)
                continue
            
            if len(linking_result.candidates) == 1:
                # No ambiguity
                linking_result.selected_candidate = linking_result.candidates[0][0]
                resolved.append(linking_result)
                continue
            
            # Score candidates by coherence with context
            best_candidate = None
            best_score = 0.0
            
            for candidate, base_score in linking_result.candidates:
                coherence_score = self._coherence_score(
                    candidate,
                    context_entities
                )
                total_score = base_score * 0.7 + coherence_score * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            
            linking_result.selected_candidate = best_candidate
            linking_result.confidence = best_score
            resolved.append(linking_result)
        
        return resolved
    
    def _coherence_score(
        self,
        candidate,
        context_entities: List[ExtractedEntity]
    ) -> float:
        """Calculate coherence score with context entities"""
        if not context_entities or candidate.name not in self.entity_graph:
            return 0.0
        
        related = set(self.entity_graph.get(candidate.name, []))
        context_names = {e.normalized_text for e in context_entities}
        
        # Score based on overlap with related entities
        overlap = related & context_names
        if not overlap:
            return 0.0
        
        return len(overlap) / len(context_names)
    
    def resolve_coreference(
        self,
        text: str,
        entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Resolve coreferences (pronouns referring to entities)"""
        import re
        
        # Simple coreference resolution
        pronouns = ['it', 'he', 'she', 'they', 'this', 'that', 'these', 'those']
        resolved = entities.copy()
        
        # Find pronouns and link to most recent entity of matching type
        last_entity_by_type = {}
        for entity in entities:
            last_entity_by_type[entity.entity_type] = entity
        
        # Pattern for pronoun detection
        for match in re.finditer(r'\b(' + '|'.join(pronouns) + r')\b', text, re.IGNORECASE):
            pronoun = match.group(1).lower()
            
            # Simple heuristic mapping
            type_mapping = {
                'it': [EntityType.TECHNOLOGY, EntityType.CONCEPT, EntityType.PRODUCT],
                'he': [EntityType.PERSON],
                'she': [EntityType.PERSON],
                'they': [EntityType.ORGANIZATION, EntityType.PERSON],
            }
            
            possible_types = type_mapping.get(pronoun, [])
            for etype in possible_types:
                if etype in last_entity_by_type:
                    # Create a reference entity
                    ref_entity = ExtractedEntity(
                        text=pronoun,
                        entity_type=etype,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.5,
                        normalized_text=last_entity_by_type[etype].normalized_text,
                        metadata={"coreference": True, "refers_to": last_entity_by_type[etype].text}
                    )
                    resolved.append(ref_entity)
                    break
        
        return resolved

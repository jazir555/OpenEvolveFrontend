"""
Natural Language Query Parser

Parses natural language queries into structured search parameters.

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
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intents"""
    SEARCH = auto()
    FIND = auto()
    HOW_TO = auto()
    WHAT_IS = auto()
    COMPARE = auto()
    RELATIONSHIP = auto()
    LIST = auto()
    COUNT = auto()
    EXPLAIN = auto()
    SUMMARIZE = auto()


@dataclass
class ParsedQuery:
    """A parsed natural language query"""
    original_query: str
    intent: QueryIntent
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    confidence: float = 1.0
    parsed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "intent": self.intent.name,
            "entities": self.entities,
            "keywords": self.keywords,
            "filters": self.filters,
            "sort_by": self.sort_by,
            "limit": self.limit,
            "confidence": self.confidence,
        }


class NaturalLanguageQueryParser:
    """
    Parses natural language queries into structured parameters
    
    Supports:
    - Intent detection
    - Entity extraction
    - Keyword extraction
    - Filter extraction (date ranges, types, etc.)
    """
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }
        self.entity_patterns = [
            r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b',  # Multi-word proper nouns
            r'\b[A-Z][a-zA-Z]+\b',  # Capitalized words
            r'"([^"]+)"',  # Quoted phrases
            r'\b\w+\.\w+\b',  # Dot notation (e.g., Class.method)
        ]
    
    def _build_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Build patterns for intent detection"""
        return {
            QueryIntent.WHAT_IS: [
                r'^what is\b', r'^what are\b', r'^define\b', r'^explain what\b'
            ],
            QueryIntent.HOW_TO: [
                r'^how (?:to|do|can|should)\b', r'^how (?:do|can) I\b', r'^steps? (?:to|for)\b'
            ],
            QueryIntent.FIND: [
                r'^find\b', r'^lookup\b', r'^search for\b', r'^where is\b', r'^locate\b'
            ],
            QueryIntent.COMPARE: [
                r'^compare\b', r'^difference between\b', r'^versus\b', r'^vs\b',
                r'^similarities between\b', r'^how does .+ compare to\b'
            ],
            QueryIntent.RELATIONSHIP: [
                r'^how (?:is|are) .+ related\b', r'^relationship between\b',
                r'^connection between\b', r'^links? between\b'
            ],
            QueryIntent.LIST: [
                r'^list\b', r'^show me\b', r'^give me\b', r'^what are all\b',
                r'^enumerate\b'
            ],
            QueryIntent.COUNT: [
                r'^how many\b', r'^count\b', r'^number of\b'
            ],
            QueryIntent.EXPLAIN: [
                r'^explain\b', r'^why\b', r'^how does\b', r'^describe how\b'
            ],
            QueryIntent.SUMMARIZE: [
                r'^summarize\b', r'^summary of\b', r'^overview of\b', r'^tl;dr\b'
            ],
        }
    
    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query"""
        query_lower = query.lower().strip()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Extract filters
        filters = self._extract_filters(query_lower)
        
        # Extract limit
        limit = self._extract_limit(query_lower)
        
        return ParsedQuery(
            original_query=query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            filters=filters,
            limit=limit,
            confidence=0.8
        )
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent from patterns"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        return QueryIntent.SEARCH
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        entities = []
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            if matches:
                # Handle tuple matches (groups)
                if isinstance(matches[0], tuple):
                    entities.extend([m[0] if len(m) > 0 else m for m in matches])
                else:
                    entities.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            e_lower = e.lower()
            if e_lower not in seen:
                seen.add(e_lower)
                unique_entities.append(e)
        
        return unique_entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove punctuation
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        # Filter stop words and short words
        keywords = [
            w for w in words
            if w not in self.stop_words and len(w) > 3
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)
        
        return unique_keywords
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from query"""
        filters = {}
        
        # Date filters
        date_patterns = [
            (r'after (\d{4}-\d{2}-\d{2})', 'created_after'),
            (r'before (\d{4}-\d{2}-\d{2})', 'created_before'),
            (r'since (\d{4})', 'year_after'),
            (r'in (\d{4})', 'year'),
        ]
        
        for pattern, key in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                filters[key] = match.group(1)
        
        # Type filters
        type_patterns = [
            (r'type:\s*(\w+)', 'node_type'),
            (r'kind:\s*(\w+)', 'node_type'),
            (r'category:\s*(\w+)', 'category'),
        ]
        
        for pattern, key in type_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                filters[key] = match.group(1)
        
        # Author filter
        author_match = re.search(r'by (\w+)', query, re.IGNORECASE)
        if author_match:
            filters['author'] = author_match.group(1)
        
        return filters
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """Extract result limit from query"""
        patterns = [
            r'top (\d+)',
            r'limit (\d+)',
            r'first (\d+)',
            r'(\d+) results?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def to_cypher(self, parsed: ParsedQuery) -> str:
        """Convert parsed query to Cypher (basic implementation)"""
        # Build MATCH clause
        match_clause = "MATCH (n)"
        
        # Build WHERE clause
        where_conditions = []
        
        if parsed.entities:
            entity_conditions = [
                f'n.name CONTAINS "{e}"'
                for e in parsed.entities[:2]  # Limit to first 2 entities
            ]
            where_conditions.append("(" + " OR ".join(entity_conditions) + ")")
        
        if parsed.keywords:
            keyword_conditions = [
                f'n.description CONTAINS "{k}"'
                for k in parsed.keywords[:3]
            ]
            if keyword_conditions:
                where_conditions.append("(" + " OR ".join(keyword_conditions) + ")")
        
        # Build filters
        if 'node_type' in parsed.filters:
            match_clause = f"MATCH (n:{parsed.filters['node_type']})"
        
        # Build query
        query_parts = [match_clause]
        
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        query_parts.append("RETURN n")
        
        if parsed.sort_by:
            query_parts.append(f"ORDER BY n.{parsed.sort_by}")
        
        if parsed.limit:
            query_parts.append(f"LIMIT {parsed.limit}")
        else:
            query_parts.append("LIMIT 10")
        
        return "\n".join(query_parts)

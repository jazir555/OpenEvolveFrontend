"""
Query Interface Module for OpenEvolve Knowledge Engine

Natural language query parsing, result formatting, caching, and
feedback loop for continuous improvement.

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

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .parser import NaturalLanguageQueryParser, ParsedQuery, QueryIntent
from .formatter import ResultFormatter, FormattedResult, OutputFormat
from .cache import QueryCache
from .feedback import FeedbackLoop, QueryFeedback


# Alias classes for compatibility with knowledge_engine/__init__.py
@dataclass
class KnowledgeQuery:
    """A knowledge query (alias for ParsedQuery)."""
    query_text: str
    intent: str = "search"
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_parsed_query(self) -> ParsedQuery:
        """Convert to ParsedQuery format."""
        return ParsedQuery(
            original_query=self.query_text,
            intent=QueryIntent.SEARCH,
            entities=self.entities,
            keywords=self.keywords,
            filters=self.filters
        )


@dataclass
class QueryResult:
    """Result of a knowledge query."""
    query: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    result_count: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_formatted_result(self, format: OutputFormat = OutputFormat.JSON) -> FormattedResult:
        """Convert to FormattedResult."""
        return FormattedResult(
            original_results=self.results,
            format=format,
            query=self.query,
            metadata=self.metadata
        )


class QueryEngine:
    """
    Main query engine for processing knowledge queries.
    """
    
    def __init__(self, cache_size: int = 1000):
        self.parser = NaturalLanguageQueryParser()
        self.formatter = ResultFormatter()
        self.cache = QueryCache(max_size=cache_size)
        self.feedback = FeedbackLoop()
    
    async def query(self, query_text: str, **kwargs) -> QueryResult:
        """
        Process a knowledge query.
        
        Args:
            query_text: The query text
            **kwargs: Additional query parameters
            
        Returns:
            QueryResult with results
        """
        start_time = datetime.utcnow()
        
        # Parse the query
        parsed = self.parser.parse(query_text)
        
        # Check cache
        cached = self.cache.get(query_text)
        if cached:
            return QueryResult(
                query=query_text,
                results=cached,
                result_count=len(cached),
                processing_time_ms=0.0,
                metadata={"cached": True}
            )
        
        # In a real implementation, this would query the knowledge graph
        # For now, return empty results
        results = []
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return QueryResult(
            query=query_text,
            results=results,
            result_count=len(results),
            processing_time_ms=processing_time,
            metadata={"parsed_intent": parsed.intent.value}
        )
    
    async def search(self, query_text: str, **kwargs) -> List[Dict[str, Any]]:
        """Simple search interface."""
        result = await self.query(query_text, **kwargs)
        return result.results


class QueryOptimizer:
    """
    Optimizes queries for better performance.
    """
    
    def optimize(self, query: KnowledgeQuery) -> KnowledgeQuery:
        """
        Optimize a knowledge query.
        
        Args:
            query: The query to optimize
            
        Returns:
            Optimized query
        """
        # In a real implementation, this would apply query optimizations
        return query
    
    def suggest_indexes(self, query_patterns: List[str]) -> List[str]:
        """
        Suggest indexes based on query patterns.
        
        Args:
            query_patterns: Common query patterns
            
        Returns:
            List of suggested indexes
        """
        return []


def create_query_engine(cache_size: int = 1000) -> QueryEngine:
    """
    Create a new query engine instance.
    
    Args:
        cache_size: Maximum cache size
        
    Returns:
        Configured QueryEngine
    """
    return QueryEngine(cache_size=cache_size)


__all__ = [
    # Parser
    'NaturalLanguageQueryParser',
    'ParsedQuery',
    'QueryIntent',
    
    # Formatter
    'ResultFormatter',
    'FormattedResult',
    'OutputFormat',
    
    # Cache
    'QueryCache',
    
    # Feedback
    'FeedbackLoop',
    'QueryFeedback',
    
    # Query Engine
    'KnowledgeQuery',
    'QueryResult',
    'QueryEngine',
    'QueryOptimizer',
    'create_query_engine',
]

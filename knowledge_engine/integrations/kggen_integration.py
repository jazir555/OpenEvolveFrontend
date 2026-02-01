"""
KG-Gen Integration for OpenEvolve Knowledge Engine (LLM-Based Implementation)

This module provides integration with the KG-Gen knowledge extraction pipeline
using LLM-based entity and relationship extraction with fallback to mock implementation
when API is not available.
"""

import asyncio
import logging
import os
import re
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import llm_utils for API calls
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from llm_utils import _request_openai_compatible_chat, _compose_messages
    LLM_UTILS_AVAILABLE = True
except ImportError:
    LLM_UTILS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraph:
    """Representation of a knowledge graph with entities and relationships."""
    entities: List[str] = None
    relations: List[Tuple[str, str, str]] = None
    entity_clusters: Dict[str, List[str]] = None  # canonical -> [duplicates]

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relations is None:
            self.relations = []
        if self.entity_clusters is None:
            self.entity_clusters = {}

    def merge(self, other: 'KnowledgeGraph'):
        """Merge another knowledge graph into this one."""
        self.entities.extend(other.entities)
        self.relations.extend(other.relations)

        # Merge entity clusters
        for canonical, duplicates in other.entity_clusters.items():
            if canonical in self.entity_clusters:
                self.entity_clusters[canonical].extend(duplicates)
            else:
                self.entity_clusters[canonical] = duplicates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'entities': self.entities,
            'relations': self.relations,
            'entity_clusters': self.entity_clusters
        }


class KGGenIntegration:
    """
    Integration with KG-Gen knowledge extraction pipeline using LLM-based extraction.

    Provides methods for:
    - Entity extraction using LLM (with mock fallback)
    - Relation extraction
    - Entity deduplication
    - Graph construction
    - Batch processing
    """

    # Default model for cost-effective extraction
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # Extraction prompt template
    EXTRACTION_PROMPT = """Extract a knowledge graph from the following text. Identify:
1. Entities with their types (PERSON, ORGANIZATION, LOCATION, TECHNOLOGY, CONCEPT, PRODUCT, EVENT)
2. Relationships between entities (as triples: subject, predicate, object)

Text: {text}

Return ONLY a JSON object in this exact format:
{{
  "entities": [{{"name": "Entity Name", "type": "TYPE"}}],
  "relations": [["Subject", "predicate", "Object"]]
}}"""

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 16000,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        test_mode: bool = False
    ):
        """
        Initialize the KG-Gen integration.

        Args:
            model: LLM model to use for extraction (default: gpt-4o-mini)
            max_tokens: Maximum tokens for model
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: API base for model access
            config: Additional configuration options
            test_mode: If True, use mock extraction instead of LLM
        """
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.config = config or self._get_default_config()
        self.test_mode = test_mode
        self._llm_available = self._check_llm_availability()

        logger.info({
            "msg": "KGGenIntegration initialized",
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "llm_available": self._llm_available,
            "test_mode": test_mode,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _check_llm_availability(self) -> bool:
        """Check if LLM extraction is available."""
        if self.test_mode:
            return False
        if not self.api_key:
            return False
        if not OPENAI_AVAILABLE and not LLM_UTILS_AVAILABLE:
            return False
        return True

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'chunk_size': 5000,
            'overlap': 200,
            'parallel_workers': 4,
            'semhash_threshold': 0.95,
            'lm_cluster_size': 128,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'clustering_algorithm': 'hdbscan',
            'min_cluster_size': 2,
            'cluster_selection_epsilon': 0.1,
            'extract_temporal': True,
            'extract_attributes': True,
            'entity_types': [
                'PERSON', 'ORGANIZATION', 'LOCATION', 'TECHNOLOGY',
                'CONCEPT', 'PRODUCT', 'EVENT', 'DATE'
            ]
        }

    async def extract_knowledge_graph(
        self,
        text: str,
        context: str = "",
        deduplication_method: str = 'FULL',
        chunk_size: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph from text using LLM-based KG-Gen implementation.

        Args:
            text: Input text to extract knowledge from
            context: Context information for extraction
            deduplication_method: Method for deduplication ('SEMHASH', 'LM_CLUSTER', 'FULL')
            chunk_size: Size of text chunks for processing
            correlation_id: Correlation ID for tracking

        Returns:
            KnowledgeGraph with extracted entities and relationships
        """
        correlation_id = correlation_id or f"kggen_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting KG-Gen extraction",
            "text_length": len(text),
            "context": context,
            "deduplication_method": deduplication_method,
            "chunk_size": chunk_size,
            "use_llm": self._llm_available,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Use LLM extraction if available, otherwise fall back to mock
            if self._llm_available:
                entities, relations = await self._extract_knowledge_llm(text)
                extraction_method = "llm"
            else:
                entities, relations = self._mock_extract_knowledge(text)
                extraction_method = "mock"

            knowledge_graph = KnowledgeGraph(
                entities=entities,
                relations=relations,
                entity_clusters={}
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "KG-Gen extraction completed",
                "correlation_id": correlation_id,
                "extraction_method": extraction_method,
                "entities_count": len(knowledge_graph.entities),
                "relations_count": len(knowledge_graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return knowledge_graph

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "KG-Gen extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return empty graph on failure
            return KnowledgeGraph()

    async def _extract_knowledge_llm(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Extract knowledge using LLM API.

        Args:
            text: Input text to extract knowledge from

        Returns:
            Tuple of (entities, relations)
        """
        # Truncate text if too long
        max_text_length = 8000
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."

        prompt = self.EXTRACTION_PROMPT.format(text=text)

        try:
            # Try using llm_utils first
            if LLM_UTILS_AVAILABLE:
                messages = _compose_messages(
                    system_message="You are a knowledge graph extraction assistant. Extract entities and relationships accurately.",
                    user_message=prompt
                )
                
                response = _request_openai_compatible_chat(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=min(self.max_tokens, 4096),
                    response_format={"type": "json_object"}
                )
            elif OPENAI_AVAILABLE:
                # Fallback to direct OpenAI client
                client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
                
                response_obj = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a knowledge graph extraction assistant. Extract entities and relationships accurately."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=min(self.max_tokens, 4096),
                    response_format={"type": "json_object"}
                )
                response = response_obj.choices[0].message.content
            else:
                # No LLM available, fall back to mock
                logger.warning("No LLM client available, falling back to mock extraction")
                return self._mock_extract_knowledge(text)

            if not response:
                logger.warning("Empty LLM response, falling back to mock extraction")
                return self._mock_extract_knowledge(text)

            # Parse JSON response
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(1))
                else:
                    logger.warning("Failed to parse LLM response as JSON, falling back to mock")
                    return self._mock_extract_knowledge(text)

            # Extract entities
            entities = []
            if "entities" in parsed and isinstance(parsed["entities"], list):
                for entity in parsed["entities"]:
                    if isinstance(entity, dict) and "name" in entity:
                        entity_str = entity["name"]
                        if "type" in entity:
                            entity_str = f"{entity['name']} ({entity['type']})"
                        entities.append(entity_str)
                    elif isinstance(entity, str):
                        entities.append(entity)

            # Extract relations
            relations = []
            if "relations" in parsed and isinstance(parsed["relations"], list):
                for rel in parsed["relations"]:
                    if isinstance(rel, list) and len(rel) >= 3:
                        relations.append((str(rel[0]), str(rel[1]), str(rel[2])))
                    elif isinstance(rel, dict):
                        # Handle dict format: {"subject": "...", "predicate": "...", "object": "..."}
                        subj = rel.get("subject") or rel.get("source") or rel.get("from")
                        pred = rel.get("predicate") or rel.get("relation") or rel.get("type")
                        obj = rel.get("object") or rel.get("target") or rel.get("to")
                        if subj and pred and obj:
                            relations.append((str(subj), str(pred), str(obj)))

            # If extraction returned no results, fall back to mock
            if not entities and not relations:
                logger.warning("LLM extraction returned empty results, falling back to mock")
                return self._mock_extract_knowledge(text)

            return entities, relations

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to mock")
            return self._mock_extract_knowledge(text)

    def _mock_extract_knowledge(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Mock knowledge extraction implementation (fallback).

        Args:
            text: Input text to extract knowledge from

        Returns:
            Tuple of (entities, relations)
        """
        # Extract potential entities (capitalized words/phrases)
        entity_pattern = r'\b[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]*){0,3}\b'
        potential_entities = list(set(re.findall(entity_pattern, text)))

        # Limit to reasonable number of entities
        entities = potential_entities[:20]  # Max 20 entities

        # Create mock relations between entities
        relations = []
        for i in range(min(len(entities), 10)):  # Max 10 relations
            if i + 1 < len(entities):
                relations.append((entities[i], "related_to", entities[i+1]))
                relations.append((entities[i], "part_of", entities[0]))  # Relate to first entity

        return entities, relations

    async def extract_from_large_document(
        self,
        document: str,
        chunk_size: int = 5000,
        overlap: int = 200,
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Extract knowledge from large documents by chunking and processing.

        Args:
            document: Large document text
            chunk_size: Size of chunks for processing
            overlap: Overlap between chunks
            correlation_id: Correlation ID for tracking

        Returns:
            Combined knowledge graph from all chunks
        """
        correlation_id = correlation_id or f"large_doc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Extracting from large document with KG-Gen",
            "document_length": len(document),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Create chunks
            chunks = self._create_chunks(document, chunk_size, overlap)

            logger.info({
                "msg": "Document chunked for processing",
                "chunk_count": len(chunks),
                "correlation_id": correlation_id
            })

            # Process each chunk and collect results
            combined_graph = KnowledgeGraph()
            successful_extractions = 0

            for i, chunk in enumerate(chunks):
                try:
                    chunk_graph = await self.extract_knowledge_graph(
                        text=chunk,
                        context=f"Chunk {i+1} of {len(chunks)}",
                        correlation_id=f"{correlation_id}_chunk_{i}"
                    )

                    # Merge the chunk graph into the combined graph
                    combined_graph.entities.extend(chunk_graph.entities)
                    combined_graph.relations.extend(chunk_graph.relations)

                    # Merge entity clusters
                    for canonical, duplicates in chunk_graph.entity_clusters.items():
                        if canonical in combined_graph.entity_clusters:
                            combined_graph.entity_clusters[canonical].extend(duplicates)
                        else:
                            combined_graph.entity_clusters[canonical] = duplicates

                    successful_extractions += 1

                except Exception as e:
                    logger.warning({
                        "msg": f"Chunk {i} extraction failed",
                        "error": str(e),
                        "correlation_id": f"{correlation_id}_chunk_{i}"
                    })
                    continue

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Large document extraction completed",
                "correlation_id": correlation_id,
                "successful_chunks": successful_extractions,
                "total_chunks": len(chunks),
                "final_entities_count": len(set(combined_graph.entities)),  # Remove duplicates
                "final_relations_count": len(combined_graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return combined_graph

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Large document extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return KnowledgeGraph()

    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create chunks from text with specified size and overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move start position by chunk_size minus overlap
            start = end - overlap

            # If we're near the end, adjust to avoid negative start
            if start >= len(text):
                break

        return chunks

    async def extract_batch(
        self,
        texts: List[str],
        correlation_id: Optional[str] = None
    ) -> List[KnowledgeGraph]:
        """
        Extract knowledge graphs from multiple texts in batch.

        Args:
            texts: List of input texts
            correlation_id: Correlation ID for tracking

        Returns:
            List of KnowledgeGraph objects
        """
        correlation_id = correlation_id or f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting batch extraction with KG-Gen",
            "text_count": len(texts),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Process all texts in parallel using asyncio
            tasks = []
            for i, text in enumerate(texts):
                task = self.extract_knowledge_graph(
                    text=text,
                    context=f"Batch item {i+1}",
                    correlation_id=f"{correlation_id}_item_{i}"
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_results = []
            successful = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Batch item {i} extraction failed",
                        "error": str(result),
                        "correlation_id": f"{correlation_id}_item_{i}"
                    })
                    processed_results.append(KnowledgeGraph())
                else:
                    processed_results.append(result)
                    if result.entities or result.relations:
                        successful += 1

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Batch extraction completed",
                "correlation_id": correlation_id,
                "successful": successful,
                "total": len(texts),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return processed_results

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Batch extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return [KnowledgeGraph() for _ in texts]

    async def deduplicate_graph(
        self,
        graph: KnowledgeGraph,
        method: str = 'FULL',
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Apply deduplication to a knowledge graph using mock implementation.

        Args:
            graph: KnowledgeGraph to deduplicate
            method: Deduplication method ('SEMHASH', 'LM_CLUSTER', 'FULL')
            correlation_id: Correlation ID for tracking

        Returns:
            Deduplicated KnowledgeGraph
        """
        correlation_id = correlation_id or f"dedup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting graph deduplication with KG-Gen",
            "entities_count": len(graph.entities),
            "relations_count": len(graph.relations),
            "method": method,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Mock deduplication - just remove duplicate entities
            unique_entities = list(set(graph.entities))
            
            # Create new graph with deduplicated entities
            result_graph = KnowledgeGraph(
                entities=unique_entities,
                relations=graph.relations,  # Relations remain the same for mock
                entity_clusters=graph.entity_clusters
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Graph deduplication completed",
                "correlation_id": correlation_id,
                "original_entities": len(graph.entities),
                "deduplicated_entities": len(result_graph.entities),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result_graph

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Graph deduplication failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return graph  # Return original graph if deduplication fails

    async def aggregate_graphs(
        self,
        graphs: List[KnowledgeGraph],
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Aggregate multiple knowledge graphs into a single graph.

        Args:
            graphs: List of KnowledgeGraph objects to aggregate
            correlation_id: Correlation ID for tracking

        Returns:
            Aggregated KnowledgeGraph
        """
        correlation_id = correlation_id or f"aggregate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting graph aggregation",
            "graph_count": len(graphs),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Aggregate all graphs by combining entities and relations
            aggregated_entities = []
            aggregated_relations = []
            aggregated_clusters = {}

            for graph in graphs:
                aggregated_entities.extend(graph.entities)
                aggregated_relations.extend(graph.relations)
                
                # Merge clusters
                for canonical, duplicates in graph.entity_clusters.items():
                    if canonical in aggregated_clusters:
                        aggregated_clusters[canonical].extend(duplicates)
                    else:
                        aggregated_clusters[canonical] = duplicates

            # Remove duplicates
            unique_entities = list(set(aggregated_entities))

            result_graph = KnowledgeGraph(
                entities=unique_entities,
                relations=aggregated_relations,
                entity_clusters=aggregated_clusters
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Graph aggregation completed",
                "correlation_id": correlation_id,
                "original_graphs": len(graphs),
                "aggregated_entities": len(result_graph.entities),
                "aggregated_relations": len(result_graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result_graph

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Graph aggregation failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return empty graph if aggregation fails
            return KnowledgeGraph()

    async def export_graph(
        self,
        graph: KnowledgeGraph,
        output_path: str,
        format: str = 'json'
    ) -> bool:
        """
        Export knowledge graph to file.

        Args:
            graph: KnowledgeGraph to export
            output_path: Path for output file
            format: Export format ('json', 'graphml', 'gexf')

        Returns:
            True if export successful
        """
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting graph export",
            "output_path": output_path,
            "format": format,
            "entities_count": len(graph.entities),
            "relations_count": len(graph.relations),
            "timestamp": start_time.isoformat()
        })

        try:
            if format.lower() == 'json':
                # Export as JSON
                graph_data = {
                    "entities": list(set(graph.entities)),  # Remove duplicates
                    "relations": graph.relations,
                    "entity_clusters": graph.entity_clusters,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "export_format": "json"
                }

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2)

                logger.info({
                    "msg": "Graph exported to JSON successfully",
                    "output_path": output_path,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                return True
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error({
                "msg": "Graph export failed",
                "output_path": output_path,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return False

    def get_kggen_status(self) -> Dict[str, Any]:
        """
        Get the status of the KG-Gen integration.

        Returns:
            Dictionary with integration status
        """
        return {
            "available": True,
            "client_initialized": self._llm_available,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "implementation": "llm" if self._llm_available else "mock",
            "llm_available": self._llm_available,
            "test_mode": self.test_mode,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing KG-Gen integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # No specific cleanup needed
        logger.info({
            "msg": "KG-Gen integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

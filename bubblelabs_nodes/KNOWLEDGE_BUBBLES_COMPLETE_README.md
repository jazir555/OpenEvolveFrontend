# Complete Knowledge Engine Bubbles for BubbleLab

## Overview

This directory contains **15 BubbleLabs integration nodes** (bubbles) that expose the full OpenEvolve Knowledge Engine functionality within BubbleLab workflows.

**Total Bubbles:** 15  
**Total Code:** ~559,000 bytes (~559 KB)  
**Status:** All bubbles verified and working

---

## Quick Reference

### Core Knowledge Operations (5 bubbles)
| Bubble | Purpose | Size |
|--------|---------|------|
| KnowledgeExtractionNode | Extract triples from text | 17 KB |
| KnowledgeQueryNode | Query knowledge graph | 21 KB |
| KnowledgeReasoningNode | Formal reasoning with Z3 | 33 KB |
| KnowledgeIntegrationNode | Multi-source integration | 24 KB |
| EntityProfileNode | Rich entity profiles | 27 KB |

### Advanced Analytics (5 bubbles)
| Bubble | Purpose | Size |
|--------|---------|------|
| TemporalKnowledgeNode | Time-based knowledge tracking | 41 KB |
| PatternMiningNode | Discover patterns with PAMI | 50 KB |
| SemanticSearchNode | Neural embedding search | 40 KB |
| CausalAnalysisNode | Causal discovery | 40 KB |
| KnowledgeEvolutionNode | Genetic algorithm optimization | 42 KB |

### Quality & Management (5 bubbles)
| Bubble | Purpose | Size |
|--------|---------|------|
| DeduplicationNode | Merge duplicate entities | 35 KB |
| ContradictionDetectionNode | Find conflicting knowledge | 66 KB |
| KnowledgeAnalyticsNode | Graph statistics & metrics | 44 KB |
| KnowledgeValidationNode | Schema & quality validation | 40 KB |
| KnowledgeImportExportNode | Import/export formats | 49 KB |

---

## Detailed Documentation

### 1. KnowledgeExtractionNode
**File:** `knowledge_extraction_node.py` (17,615 bytes)

Extract structured knowledge from unstructured text using multiple NLP strategies.

**Operations:**
- Extract entities, relationships, and triples
- Support for DeepKE, OneKE, KG-Gen extractors
- Auto-extractor selection

**Config:** `extractor`, `min_confidence`, `include_metadata`, `domain`

**Input:** `{ "text": "OpenAI released GPT-4..." }`

**Output:** `{ "triples": [...], "entities": [...], "confidence": 0.92 }`

---

### 2. KnowledgeQueryNode
**File:** `knowledge_query_node.py` (21,431 bytes)

Query the unified knowledge graph for entities, relationships, and paths.

**Operations:**
- Query triples by S/P/O filters
- Find paths between entities
- Get entity neighborhoods
- Export knowledge graph

**Config:** `query_type`, `subject`, `predicate`, `object`, `min_confidence`, `max_path_length`, `depth`

**Input:** Query specification based on query_type

**Output:** Query results with triples, paths, or exported data

---

### 3. KnowledgeReasoningNode
**File:** `knowledge_reasoning_node.py` (33,068 bytes)

Verify, validate, and reason over knowledge using formal methods.

**Operations:**
- Verify consistency with Z3
- Detect contradictions
- Infer new facts
- Validate logical statements

**Config:** `reasoning_type`, `premises`, `conclusion`, `include_explanation`, `timeout`

**Input:** `{ "premises": [...], "conclusion": "..." }`

**Output:** `{ "valid": true, "explanation": "...", "inferred": [...] }`

---

### 4. KnowledgeIntegrationNode
**File:** `knowledge_integration_node.py` (23,614 bytes)

Integrate knowledge from multiple sources using the Unified KG Integration Hub.

**Operations:**
- Initialize knowledge hub
- Extract with multiple extractors (parallel)
- Merge knowledge sources
- Export integrated knowledge
- Health check all integrations

**Config:** `operation`, `extractors`, `export_format`, `enable_reasoning`, `enable_temporal`

---

### 5. EntityProfileNode
**File:** `entity_profile_node.py` (26,643 bytes)

Create and manage rich entity profiles with relationships and provenance.

**Operations:**
- Create entity profiles
- Update existing profiles
- Get profile by name
- Merge multiple profiles

**Config:** `operation`, `entity_name`, `entity_types`, `properties`, `relationships`, `confidence`

---

### 6. TemporalKnowledgeNode
**File:** `temporal_knowledge_node.py` (40,859 bytes)

Track and query knowledge changes over time with temporal awareness.

**Operations:**
- Store knowledge with timestamps
- Query knowledge at specific time
- Track knowledge evolution
- Compare time periods
- Get entity history

**Config:** `operation`, `timestamp`, `valid_from`, `valid_until`, `entity_id`, `time_window_days`

**Input:** Temporal specifications

**Output:** Time-valid knowledge, change history, period comparisons

---

### 7. PatternMiningNode
**File:** `pattern_mining_node.py` (50,213 bytes)

Discover patterns, associations, and anomalies in knowledge graphs using PAMI.

**Operations:**
- Mine frequent patterns
- Discover association rules (if-then)
- Find sequential/temporal patterns
- Detect anomalies

**Config:** `mining_type`, `min_support`, `min_confidence`, `max_pattern_length`, `entity_types`

**Output:** `{ "patterns": [...], "rules": [...], "anomalies": [...], "statistics": {...} }`

---

### 8. SemanticSearchNode
**File:** `semantic_search_node.py` (40,129 bytes)

Find similar entities and search knowledge using neural embeddings.

**Operations:**
- Generate embeddings
- Find semantically similar entities
- Natural language search
- Cluster entities
- Recommend related entities

**Config:** `operation`, `query`, `top_k`, `similarity_threshold`, `entity_types`, `embedding_model`

**Output:** Similar entities, clusters, recommendations with similarity scores

---

### 9. CausalAnalysisNode
**File:** `causal_analysis_node.py` (40,010 bytes)

Discover causal relationships and build causal graphs from knowledge.

**Operations:**
- Discover causal relationships (PC, FCI, GES, LiNGAM algorithms)
- Build causal graphs
- Identify confounding variables
- Estimate causal effects
- Validate causal hypotheses

**Config:** `operation`, `variables`, `target_variable`, `treatment_variable`, `algorithm`, `significance_level`

**Output:** `{ "causal_graph": {...}, "relationships": [...], "effects": [...], "confounders": [...] }`

---

### 10. KnowledgeEvolutionNode
**File:** `knowledge_evolution_node.py` (42,315 bytes)

Evolve and optimize knowledge using genetic algorithms.

**Operations:**
- Evolve knowledge through generations
- Optimize knowledge structures
- Select best variants
- Mutate and crossover
- Track fitness improvements

**Config:** `operation`, `generations`, `population_size`, `mutation_rate`, `crossover_rate`, `fitness_metric`, `selection_strategy`

**Output:** `{ "evolved_knowledge": {...}, "fitness_history": [...], "improvements": [...] }`

---

### 11. DeduplicationNode
**File:** `deduplication_node.py` (35,451 bytes)

Find and merge duplicate entities and triples in knowledge graphs.

**Operations:**
- Find duplicate entities (fuzzy matching)
- Find duplicate triples
- Merge duplicates with conflict resolution
- Auto-deduplication
- Generate reports

**Config:** `operation`, `similarity_threshold`, `entity_types`, `merge_strategy`, `auto_merge`

**Output:** `{ "duplicates": [...], "merged": [...], "conflicts": [...], "report": {...} }`

---

### 12. ContradictionDetectionNode
**File:** `contradiction_detection_node.py` (65,954 bytes)

Detect and resolve conflicting knowledge in the graph.

**Operations:**
- Detect contradictions (logical, temporal, factual, semantic)
- Analyze contradiction severity
- Suggest resolutions
- Apply resolution strategies
- Generate reports

**Config:** `operation`, `severity_threshold`, `check_types`, `entity_scope`, `resolution_strategy`

**Output:** `{ "contradictions": [...], "severity_counts": {...}, "resolutions": [...] }`

---

### 13. KnowledgeAnalyticsNode
**File:** `knowledge_analytics_node.py` (43,684 bytes)

Generate statistics, metrics, and analytics for knowledge graphs.

**Operations:**
- Graph statistics (nodes, edges, density)
- Centrality metrics (degree, betweenness, pagerank)
- Entity distribution analysis
- Quality metrics
- Growth/change metrics

**Config:** `analysis_type`, `metrics`, `entity_types`, `time_range`, `compare_with_previous`, `export_format`

**Output:** `{ "statistics": {...}, "centrality": {...}, "report": {...} }`

---

### 14. KnowledgeValidationNode
**File:** `knowledge_validation_node.py` (40,182 bytes)

Validate knowledge quality, completeness, and schema compliance.

**Operations:**
- Schema validation
- Completeness checking
- Quality assessment
- Reference validation
- Format validation

**Config:** `validation_type`, `schema_id`, `required_properties`, `quality_threshold`, `strict_mode`

**Output:** `{ "valid": true/false, "score": 0.92, "errors": [...], "warnings": [...] }`

---

### 15. KnowledgeImportExportNode
**File:** `knowledge_import_export_node.py` (48,519 bytes)

Import and export knowledge in multiple formats.

**Operations:**
- Export to JSON, RDF/TTL, CSV, N-Quads, NetworkX
- Import from various formats
- Transform between formats
- Validate during import
- Handle compression (gzip, zip)

**Config:** `operation`, `format`, `source_path`, `destination_path`, `merge_strategy`, `compression`

**Output:** `{ "success": true, "records_processed": 100, "file_path": "..." }`

---

## Workflow Examples

### Example 1: Comprehensive Research Analysis
```
[Document] → [KnowledgeExtractionNode]
    → [TemporalKnowledgeNode: store with timestamp]
    → [KnowledgeValidationNode: validate]
    → [ContradictionDetectionNode: check conflicts]
    → [DeduplicationNode: merge duplicates]
    → [KnowledgeReasoningNode: verify logic]
    → [PatternMiningNode: find patterns]
    → [KnowledgeAnalyticsNode: generate stats]
    → [Export]
```

### Example 2: Knowledge Base Maintenance
```
[Existing KG] → [ContradictionDetectionNode: detect conflicts]
    → [DeduplicationNode: merge duplicates]
    → [KnowledgeValidationNode: quality check]
    → [KnowledgeAnalyticsNode: assess health]
    → [KnowledgeEvolutionNode: optimize]
    → [TemporalKnowledgeNode: version control]
```

### Example 3: Semantic Discovery
```
[Query: "AI labs in California"] → [SemanticSearchNode: find similar]
    → [KnowledgeQueryNode: get details]
    → [CausalAnalysisNode: find influences]
    → [PatternMiningNode: find common patterns]
    → [EntityProfileNode: enrich profiles]
```

### Example 4: Data Integration Pipeline
```
[Multiple Sources] → [KnowledgeImportExportNode: import]
    → [KnowledgeIntegrationNode: merge]
    → [DeduplicationNode: deduplicate]
    → [KnowledgeValidationNode: validate]
    → [KnowledgeAnalyticsNode: analyze]
    → [KnowledgeImportExportNode: export to RDF]
```

---

## Implementation Details

### Base Class
All nodes inherit from `BubbleLabsNode`:
```python
class MyNode(BubbleLabsNode):
    DISPLAY_NAME = "My Node"
    DESCRIPTION = "What it does"
    ICON = "my-icon"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"
    
    def execute(self, inputs, context):
        # Implementation
        pass
```

### Safe Import Pattern
```python
SomeClass = self.safe_import(
    'module.path.SomeClass',
    error_msg="Description if unavailable"
)
```

### Error Handling
```python
raise NodeExecutionError(
    node_name=self.__class__.__name__,
    message="What went wrong",
    details={"context": "additional info"}
)
```

---

## Testing

### Verify All Bubbles
```bash
cd knowledge_engine
python verify_all_bubbles.py
```

### Run Demo
```bash
cd knowledge_engine/examples
python complete_workflow_demo.py
```

---

## Category Distribution

| Category | Count | Bubbles |
|----------|-------|---------|
| Extraction | 1 | KnowledgeExtractionNode |
| Query | 1 | KnowledgeQueryNode |
| Reasoning | 2 | KnowledgeReasoningNode, CausalAnalysisNode |
| Integration | 2 | KnowledgeIntegrationNode, KnowledgeImportExportNode |
| Management | 4 | EntityProfileNode, DeduplicationNode, TemporalKnowledgeNode, KnowledgeEvolutionNode |
| Analytics | 3 | PatternMiningNode, SemanticSearchNode, KnowledgeAnalyticsNode |
| Quality | 2 | ContradictionDetectionNode, KnowledgeValidationNode |

---

## Dependencies

### Required
- `knowledge_engine` - Core knowledge engine
- `bubblelabs_nodes/base_node.py` - Base class

### Optional (with graceful fallbacks)
- **Extraction:** `deepke`, `oneke`, `kg_gen`
- **Reasoning:** `z3`, `causal-learn`
- **Temporal:** `graphiti`
- **Neural:** `neuralkg`, `sentence-transformers`
- **Mining:** `pami`, `karateclub`
- **Evolution:** `openevolve`

---

## Version History

### v1.0.0 (2026-02-01)
- Initial release
- 15 knowledge engine bubbles
- Full BubbleLab integration
- Safe imports with fallbacks for all dependencies
- Comprehensive documentation

---

## Support

For issues or questions:
- Check `SYSTEM_ARCHITECTURE_AND_USAGE_GUIDE.md` in knowledge_engine
- Review example workflows in `examples/`
- Run diagnostic: `python verify_all_bubbles.py`

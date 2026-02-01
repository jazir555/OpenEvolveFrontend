# Knowledge Graph Integration Specification

## Document Information
- **Version**: 1.0
- **Date**: February 1, 2026
- **Status**: Draft
- **Authors**: OpenEvolve Team

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Graph Schema](#graph-schema)
4. [Integration Patterns](#integration-patterns)
5. [Data Flow](#data-flow)
6. [Storage Backends](#storage-backends)
7. [Query Language](#query-language)
8. [Performance](#performance)
9. [Security](#security)
10. [Monitoring](#monitoring)

## Overview

### Purpose
This document specifies the knowledge graph integration architecture for the OpenEvolve ecosystem. It defines how multiple knowledge graph systems (DeepKE, OneKE, KG-Gen, Graphiti, etc.) integrate and work together to provide a unified knowledge representation.

### Goals
- Define unified graph schema across multiple systems
- Establish integration patterns for different knowledge sources
- Specify query and traversal mechanisms
- Ensure consistency across graph systems
- Enable real-time updates and temporal tracking

### Non-Goals
- Specifying internal implementation of individual graph systems
- Defining UI components
- Detailing specific business logic outside of graph operations

## Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   OpenEvolve    │    │  Graph Integration   │    │  Knowledge      │
│                 │◄──►│  Layer              │◄──►│  Graphs         │
│  Evolution      │    │                     │    │                 │
│  Process        │    │  • Schema Manager   │    │  • Neo4j        │
│                 │    │  • Query Router     │    │  • Memgraph     │
│  • Controllers  │    │  • Triple Store     │    │  • Qdrant       │
│  • Evaluators   │    │  • Temporal Layer   │    │  • PostgreSQL   │
│  • Database     │    │  • Consistency      │    │  • Custom       │
└─────────────────┘    │    Manager         │    └─────────────────┘
                       └──────────────────────┘
                                    ▲
                       ┌──────────────────────┐
                       │  Graph Analytics     │
                       │                     │
                       │  • Community Det.   │
                       │  • Centrality       │
                       │  • Path Analysis    │
                       │  • Embeddings       │
                       └──────────────────────┘
```

### Component Roles
- **Schema Manager**: Defines unified schema across systems
- **Query Router**: Routes queries to appropriate backends
- **Triple Store**: Normalizes and stores knowledge triples
- **Temporal Layer**: Handles time-based queries and evolution
- **Consistency Manager**: Ensures data consistency across systems

## Graph Schema

### Core Node Types
```cypher
// Project-related nodes
:Project { id, name, description, domain, created_at, updated_at, stage }

// Artifact-related nodes  
:Artifact { id, type, content_hash, created_at, status }

// Code-related nodes
:CodeEntity { id, name, type, language, complexity, location }
:Function { id, name, signature, complexity, language }
:Class { id, name, inheritance, language }
:Module { id, name, language, path }

// Knowledge-related nodes
:Concept { id, name, definition, domain, confidence }
:Technique { id, name, description, category, effectiveness }
:Pattern { id, name, description, applicability, examples }
:Algorithm { id, name, complexity, type, description }

// Evolution-related nodes
:EvolutionStep { id, iteration, fitness_score, timestamp }
:Improvement { id, type, description, impact_score }
```

### Core Relationship Types
```cypher
// Project hierarchy
(Project)-[:HAS_ARTIFACT]->(Artifact)
(Artifact)-[:GENERATES_INSIGHT]->(Insight)

// Code relationships
(Function)-[:BELONGS_TO]->(Class)
(Class)-[:BELONGS_TO]->(Module)
(CodeEntity)-[:CALLS]->(CodeEntity)
(CodeEntity)-[:USES]->(Concept)

// Knowledge relationships
(Concept)-[:RELATED_TO]->(Concept)
(Concept)-[:INSTANCE_OF]->(Concept)
(Technique)-[:IMPLEMENTS]->(Algorithm)
(Pattern)-[:USES]->(Concept)
(Algorithm)-[:OPTIMIZES]->(PerformanceMetric)

// Evolution relationships
(EvolutionStep)-[:IMPROVES]->(CodeEntity)
(EvolutionStep)-[:DISCOVERED]->(Technique)
(Improvement)-[:APPLIES_TO]->(CodeEntity)
(Artifact)-[:CONTAINS]->(CodeEntity)

// Temporal relationships
(Concept)-[:EVOLVED_FROM {timestamp: DateTime}]->(Concept)
```

### Schema Constraints
```cypher
// Uniqueness constraints
CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT artifact_id_unique IF NOT EXISTS FOR (a:Artifact) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE;

// Existence constraints
CREATE CONSTRAINT concept_name_exists IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS NOT NULL;
CREATE CONSTRAINT project_name_exists IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS NOT NULL;
```

## Integration Patterns

### 1. Triple Normalization Pattern
**Purpose**: Normalize knowledge from different sources into consistent triples

**Implementation**:
1. Extract knowledge from source system
2. Map to canonical schema
3. Validate against constraints
4. Store in unified triple store

**Example**:
```python
class TripleNormalizer:
    def normalize_from_deepke(self, deepke_output):
        # Convert DeepKE format to canonical triples
        normalized = []
        for entity in deepke_output.entities:
            normalized.append(KnowledgeTriple(
                subject=entity.text,
                predicate="instance_of",
                object=entity.type,
                confidence=entity.confidence,
                source="deepke"
            ))
        return normalized
    
    def normalize_from_oneke(self, oneke_output):
        # Convert OneKE format to canonical triples
        normalized = []
        for triple in oneke_output.triples:
            normalized.append(KnowledgeTriple(
                subject=triple.head,
                predicate=triple.relation,
                object=triple.tail,
                confidence=triple.confidence,
                source="oneke"
            ))
        return normalized
```

### 2. Schema Mapping Pattern
**Purpose**: Map different knowledge graph schemas to unified schema

**Implementation**:
1. Define mapping rules for each source
2. Transform nodes and relationships
3. Validate transformed data
4. Merge with existing graph

**Example Mapping**:
```yaml
# DeepKE to Canonical mapping
deepke_mapping:
  entity_types:
    Person: Concept
    Organization: Concept
    Location: Concept
  relation_types:
    LocatedIn: RELATED_TO
    FoundedBy: RELATED_TO
    WorksFor: RELATED_TO

# OneKE to Canonical mapping
oneke_mapping:
  entity_types:
    Algorithm: Algorithm
    Technique: Technique
    Pattern: Pattern
  relation_types:
    implements: IMPLEMENTS
    uses: USES
    improves: IMPROVES
```

### 3. Temporal Consistency Pattern
**Purpose**: Maintain temporal consistency across evolution steps

**Implementation**:
1. Track timestamp for all changes
2. Maintain version history
3. Support time-travel queries
4. Handle concurrent updates

**Example**:
```python
class TemporalGraphManager:
    def update_with_timestamp(self, node_id, properties, timestamp):
        # Create new version with timestamp
        new_node = self.create_version(node_id, properties, timestamp)
        
        # Link to previous version
        prev_version = self.get_latest_version(node_id)
        if prev_version:
            self.create_relationship(
                prev_version, 
                "REPLACED_BY", 
                new_node, 
                {"timestamp": timestamp}
            )
        
        return new_node
```

## Data Flow

### Ingestion Pipeline
```
External Source (DeepKE, OneKE, KG-Gen) 
         ↓
   Raw Data Extraction
         ↓
   Schema Mapping & Normalization
         ↓
   Validation & Constraint Checking
         ↓
   Deduplication & Merging
         ↓
   Temporal Versioning
         ↓
   Storage in Triple Store
         ↓
   Indexing & Optimization
```

### Query Resolution
```
Incoming Query
         ↓
   Query Parser & Validator
         ↓
   Schema Translation
         ↓
   Backend Selection (Neo4j/Memgraph/Qdrant)
         ↓
   Query Execution
         ↓
   Result Aggregation
         ↓
   Consistency Verification
         ↓
   Response Formatting
```

### Real-time Updates
```
Knowledge Source Change
         ↓
   Change Detection
         ↓
   Event Publishing
         ↓
   Schema Validation
         ↓
   Graph Update Transaction
         ↓
   Temporal Version Creation
         ↓
   Index Updates
         ↓
   Cache Invalidation
         ↓
   Subscriber Notifications
```

## Storage Backends

### Primary Storage: Neo4j/Memgraph
**Use Case**: Complex graph traversals, relationship queries
**Configuration**:
```yaml
neo4j_config:
  uri: "bolt://graph-db:7687"
  username: "graph_user"
  password: "${GRAPH_DB_PASSWORD}"
  max_connection_lifetime: 30m
  max_connection_pool_size: 50
  connection_acquisition_timeout: 2m
```

### Vector Storage: Qdrant
**Use Case**: Semantic similarity, embedding-based search
**Configuration**:
```yaml
qdrant_config:
  host: "vector-db"
  port: 6333
  collection_name: "knowledge_embeddings"
  vector_size: 768
  distance: "Cosine"
```

### Relational Storage: PostgreSQL
**Use Case**: Metadata, structured data, temporal sequences
**Configuration**:
```yaml
postgresql_config:
  host: "metadata-db"
  port: 5432
  database: "knowledge_metadata"
  username: "metadata_user"
  password: "${METADATA_DB_PASSWORD}"
  pool_size: 20
```

### Cache Layer: Redis
**Use Case**: Query results, frequently accessed nodes
**Configuration**:
```yaml
redis_config:
  host: "cache"
  port: 6379
  db: 0
  max_memory: "2gb"
  ttl_seconds: 3600
```

## Query Language

### Supported Languages
1. **Cypher** (Neo4j/Memgraph)
2. **GraphQL** (Unified interface)
3. **SPARQL** (RDF triple stores)
4. **Custom DSL** (Simplified for applications)

### GraphQL Schema
```graphql
type Query {
  # Project queries
  project(id: ID!): Project
  projects(filter: ProjectFilter): [Project!]!
  
  # Knowledge queries
  concept(name: String!): Concept
  concepts(filter: ConceptFilter, limit: Int): [Concept!]!
  
  # Code queries
  codeEntity(id: ID!): CodeEntity
  functions(filter: FunctionFilter): [Function!]!
  
  # Evolution queries
  evolutionSteps(projectId: ID!, limit: Int): [EvolutionStep!]!
  
  # Path queries
  paths(start: ID!, end: ID!, maxDepth: Int): [[Node!]!]!
}

type Mutation {
  # Create/update operations
  createProject(input: ProjectInput!): Project!
  createConcept(input: ConceptInput!): Concept!
  createRelationship(input: RelationshipInput!): Boolean!
}

# Example query
query GetProjectKnowledge($projectId: ID!) {
  project(id: $projectId) {
    id
    name
    artifacts {
      id
      type
      insights {
        id
        type
        description
        confidence
      }
    }
    relatedConcepts(limit: 10) {
      id
      name
      definition
      connectedFunctions {
        id
        name
        complexity
      }
    }
  }
}
```

### Cypher Examples
```cypher
// Find all optimization techniques for a specific algorithm
MATCH (alg:Algorithm {name: $algorithmName})<-[:IMPLEMENTS]-(tech:Technique)
WHERE tech.effectiveness > 0.7
RETURN tech.name, tech.description, tech.effectiveness
ORDER BY tech.effectiveness DESC

// Find evolution path for a code entity
MATCH path = (start:CodeEntity {id: $entityId})-[:EVOLVED_TO*]->(end:CodeEntity)
WHERE length(path) <= 5
RETURN nodes(path) AS evolution_path, relationships(path) AS changes

// Find related concepts within domain
MATCH (c:Concept {name: $seedConcept})-[:RELATED_TO*1..3]-(related:Concept)
WHERE c.domain = $domain AND related.domain = $domain
RETURN DISTINCT related.name, related.definition
```

## Performance

### Indexing Strategy
```cypher
// Node property indexes
CREATE INDEX idx_project_name FOR (p:Project) ON (p.name);
CREATE INDEX idx_concept_name FOR (c:Concept) ON (c.name);
CREATE INDEX idx_artifact_type FOR (a:Artifact) ON (a.type);
CREATE INDEX idx_timestamp FOR (n) ON (n.created_at);

// Text search indexes
CREATE TEXT INDEX idx_concept_definition FOR (c:Concept) ON (c.definition);
CREATE TEXT INDEX idx_technique_desc FOR (t:Technique) ON (t.description);

// Composite indexes
CREATE INDEX idx_project_stage_status FOR (p:Project) ON (p.stage, p.status);
```

### Query Optimization
- **Query Plans**: Cache execution plans for frequent queries
- **Result Caching**: Cache results for read-heavy operations
- **Batch Operations**: Use batch APIs for bulk updates
- **Connection Pooling**: Maintain connection pools to databases

### Performance Targets
- **Query Response Time**: <100ms for simple queries, <500ms for complex
- **Ingestion Rate**: 1000+ triples/second
- **Concurrent Users**: 1000+ simultaneous connections
- **Graph Size**: Support 10M+ nodes and 100M+ relationships

## Security

### Access Control
- **Row-level security**: Project-based data isolation
- **Attribute-based control**: Field-level access restrictions
- **API key scopes**: Granular permissions per key
- **Role-based access**: Predefined roles for different user types

### Data Protection
- **Encryption at rest**: All graph data encrypted
- **Encryption in transit**: TLS 1.3 for all connections
- **PII Detection**: Automatic detection and masking of sensitive data
- **Audit Logging**: Complete trail of all graph operations

### Security Headers
```python
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY", 
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'"
}
```

## Monitoring

### Metrics Collection
- **Query Performance**: Response times, execution counts
- **System Health**: Database connectivity, resource usage
- **Data Quality**: Constraint violations, data completeness
- **User Activity**: API usage, popular queries

### Logging Standards
```json
{
  "timestamp": "2026-02-01T12:00:00Z",
  "level": "INFO",
  "service": "graph-integration",
  "operation": "query_execution",
  "query_type": "path_traversal",
  "execution_time_ms": 150,
  "result_count": 25,
  "user_id": "usr_123",
  "project_id": "proj_456",
  "query_hash": "a1b2c3d4e5f6"
}
```

### Health Checks
- **Database connectivity**: Check all storage backends
- **Index health**: Verify index integrity
- **Constraint validation**: Check for constraint violations
- **Performance thresholds**: Monitor for degradation

### Alerting
- **Critical**: Database downtime, constraint violations
- **Warning**: Performance degradation, high error rates
- **Info**: Maintenance windows, planned updates

## Appendix

### Glossary
- **Triple**: Subject-Predicate-Object relationship in RDF format
- **Node**: Entity in the knowledge graph
- **Relationship**: Connection between nodes
- **Schema**: Definition of node types and relationship types
- **Temporal Graph**: Graph with time-based properties and relationships

### References
- Neo4j Cypher Manual
- GraphQL Specification
- RDF 1.1 Concepts and Abstract Syntax
- Property Graph Semantics

### Change Log
- **v1.0** - Initial specification
// Neo4j Initialization Script
// OpenEvolve Knowledge Engine - Phase 1.1.1
//
// This script sets up the database schema for the knowledge graph with:
// - Vector indices for semantic search
// - Temporal capabilities for time-based queries
// - Constraint enforcement for data integrity
// - Performance optimizations
//
// Usage: Run this script after Neo4j starts up
// cypher-shell -u neo4j -p <password> < init_neo4j.cypher

// ============================================================================
// SECTION 1: DATABASE CONFIGURATION
// ============================================================================

// Show current database and version
CALL dbms.components() YIELD name, versions, edition
RETURN name, versions[0] as version, edition;

// ============================================================================
// SECTION 2: CONSTRAINTS (Data Integrity)
// ============================================================================

// Ensure unique node IDs for entities
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Ensure unique node IDs for documents
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Ensure unique node IDs for concepts
CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

// Ensure unique node IDs for relationships
CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS
FOR (r:Relationship) REQUIRE r.id IS UNIQUE;

// Ensure uniqueness on entity names (for deduplication)
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

// Ensure uniqueness on document URIs
CREATE CONSTRAINT document_uri_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.uri IS UNIQUE;

// ============================================================================
// SECTION 3: INDICES (Performance Optimization)
// ============================================================================

// Index for entity name lookups
CREATE INDEX entity_name_idx IF NOT EXISTS
FOR (e:Entity) ON (e.name);

// Index for entity type lookups
CREATE INDEX entity_type_idx IF NOT EXISTS
FOR (e:Entity) ON (e.type);

// Index for document timestamps
CREATE INDEX document_created_idx IF NOT EXISTS
FOR (d:Document) ON (d.created_at);

// Index for document metadata
CREATE INDEX document_metadata_idx IF NOT EXISTS
FOR (d:Document) ON (d.source);

// Index for concept lookups
CREATE INDEX concept_name_idx IF NOT EXISTS
FOR (c:Concept) ON (c.name);

// Index for relationship lookups
CREATE INDEX relationship_type_idx IF NOT EXISTS
FOR (r:Relationship) ON (r.type);

// Index for temporal queries
CREATE INDEX entity_valid_from_idx IF NOT EXISTS
FOR (e:Entity) ON (e.valid_from);

CREATE INDEX entity_valid_to_idx IF NOT EXISTS
FOR (e:Entity) ON (e.valid_to);

// Full-text search index for entity content
CREATE FULLTEXT INDEX entity_content_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.description, e.content];

// Full-text search index for documents
CREATE FULLTEXT INDEX document_content_fulltext IF NOT EXISTS
FOR (d:Document) ON EACH [d.title, d.content];

// ============================================================================
// SECTION 4: VECTOR INDICES (Semantic Search)
// ============================================================================

// Note: Vector indices require Neo4j 5.26+ with GDS library installed
// These indices enable efficient similarity search using embeddings

// Vector index for entity embeddings (using GDS library)
// This enables semantic search over entities
CALL db.index.vector.createNodeIndex(
    'entity_embeddings',
    'Entity',
    'embedding',
    {
        dimension: 1536,  // Adjust based on your embedding model
        similarityFunction: 'cosine'
    }
) YIELD indexName, indexType, entityType, properties
RETURN indexName, indexType, entityType, properties;

// Vector index for document embeddings
CALL db.index.vector.createNodeIndex(
    'document_embeddings',
    'Document',
    'embedding',
    {
        dimension: 1536,
        similarityFunction: 'cosine'
    }
) YIELD indexName, indexType, entityType, properties
RETURN indexName, indexType, entityType, properties;

// Vector index for concept embeddings
CALL db.index.vector.createNodeIndex(
    'concept_embeddings',
    'Concept',
    'embedding',
    {
        dimension: 1536,
        similarityFunction: 'cosine'
    }
) YIELD indexName, indexType, entityType, properties
RETURN indexName, indexType, entityType, properties;

// ============================================================================
// SECTION 5: TEMPORAL CONFIGURATION
// ============================================================================

// Temporal nodes for tracking time-based validity
CREATE CONSTRAINT temporal_id_unique IF NOT EXISTS
FOR (t:Temporal) REQUIRE t.id IS UNIQUE;

// Index for temporal queries
CREATE INDEX temporal_valid_from_idx IF NOT EXISTS
FOR (t:Temporal) ON (t.valid_from);

CREATE INDEX temporal_valid_to_idx IF NOT EXISTS
FOR (t:Temporal) ON (t.valid_to);

// ============================================================================
// SECTION 6: SAMPLE DATA STRUCTURES (Optional - for testing)
// ============================================================================

// Create a sample entity with temporal metadata
// MERGE (e:Entity {
//     id: 'entity-001',
//     name: 'Sample Entity',
//     type: 'Concept',
//     description: 'A sample entity for testing',
//     created_at: datetime(),
//     updated_at: datetime(),
//     valid_from: datetime(),
//     valid_to: datetime() + duration('P1Y')
// });

// ============================================================================
// SECTION 7: DATABASE STATISTICS
// ============================================================================

// Show all constraints
CALL db.constraints() YIELD description
RETURN description;

// Show all indexes
CALL db.indexes() YIELD indexName, indexType, entityType, properties, state
RETURN indexName, indexType, entityType, properties, state;

// Show database statistics
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Primitive count") YIELD attributes
RETURN attributes;

// ============================================================================
// SECTION 8: VERIFICATION
// ============================================================================

// Verify constraints were created
CALL db.constraints() YIELD description
WHERE description CONTAINS 'entity_id_unique'
   OR description CONTAINS 'document_id_unique'
   OR description CONTAINS 'concept_id_unique'
RETURN description;

// Verify indexes were created
CALL db.indexes() YIELD indexName, indexType, state
RETURN indexName, indexType, state;

// ============================================================================
// SECTION 9: UTILITY PROCEDURES
// ============================================================================

// Create a procedure to clear all data (use with caution!)
// This should only be used in development environments
// CREATE PROCEDURE clearAllData()
// BEGIN
//     MATCH (n) DETACH DELETE n;
//     RETURN 'All data cleared' as result;
// END;

// Create a procedure to get database statistics
// CREATE PROCEDURE getDatabaseStats()
// BEGIN
//     MATCH (n) WITH count(n) as nodeCount
//     MATCH ()-[r]->() WITH nodeCount, count(r) as relCount
//     RETURN nodeCount, relCount;
// END;

// ============================================================================
// END OF INITIALIZATION SCRIPT
// ============================================================================

RETURN 'Neo4j database initialized successfully!' as message;

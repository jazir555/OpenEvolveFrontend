#!/bin/bash
##############################################################################
# Graphiti Entity/Relationship CRUD Probe Script
#
# Purpose: Verify full CRUD operations for entities and relationships
# Compliance: Law of Idempotency - operations safe to run multiple times
#
# Environment Variables Required:
#   NEO4J_URI           - Neo4j connection URI (default: bolt://localhost:7687)
#   NEO4J_USER          - Neo4j username (default: neo4j)
#   NEO4J_PASSWORD      - Neo4j password (required)
#   TIMEOUT_MS          - Request timeout in milliseconds (default: 10000)
#
# Exit Codes:
#   0 - All probes passed
#   1 - Required environment variable missing
#   2 - Entity create failed
#   3 - Entity read failed
#   4 - Entity update failed
#   5 - Entity delete failed
#   6 - Relationship operations failed
#   7 - cypher-shell not available
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
TIMEOUT_MS="${TIMEOUT_MS:-10000}"

# Test entity UUID
TEST_ENTITY_UUID="probe_entity_$(date +%s)"
TEST_EDGE_UUID="probe_edge_$(date +%s)"

# =============================================================================
# Utility Functions
# =============================================================================

log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_entities.sh\"}"
}

check_cypher_shell() {
    if ! command -v cypher-shell &> /dev/null; then
        log_json "error" "cypher-shell is not installed or not in PATH"
        exit 7
    fi
}

cypher_query() {
    local query="$1"
    echo "$query" | cypher-shell \
        -a "$NEO4J_URI" \
        -u "$NEO4J_USER" \
        -p "$NEO4J_PASSWORD" \
        --format plain \
        2>&1 || echo "ERROR: Query failed"
}

cleanup_test_data() {
    log_json "info" "Cleaning up test data"

    cypher_query "
        MATCH (e:Entity {uuid: '$TEST_ENTITY_UUID'})
        DETACH DELETE e;
    " > /dev/null 2>&1 || true
}

# =============================================================================
# Probe Functions - CRUD Operations
# =============================================================================

# CREATE: Test entity creation
probe_entity_create() {
    log_json "info" "Testing CREATE: Create new entity"

    local query="
        CREATE (e:Entity {
            uuid: '$TEST_ENTITY_UUID',
            name: 'ProbeTestEntity',
            labels: ['Test', 'Probe'],
            summary: 'Test entity for CRUD probe',
            created_at: datetime()
        })
        RETURN e.uuid AS uuid, e.name AS name;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Entity CREATE failed: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "$TEST_ENTITY_UUID"; then
        log_json "error" "Entity CREATE returned unexpected UUID: $response"
        return 1
    fi

    log_json "info" "Entity CREATE successful - UUID: $TEST_ENTITY_UUID"

    return 0
}

# READ: Test entity retrieval
probe_entity_read() {
    log_json "info" "Testing READ: Retrieve entity by UUID"

    local query="
        MATCH (e:Entity {uuid: '$TEST_ENTITY_UUID'})
        RETURN e.uuid AS uuid, e.name AS name, e.summary AS summary;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Entity READ failed: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "$TEST_ENTITY_UUID"; then
        log_json "error" "Entity READ did not find entity: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "ProbeTestEntity"; then
        log_json "error" "Entity READ missing expected name: $response"
        return 1
    fi

    log_json "info" "Entity READ successful"

    return 0
}

# UPDATE: Test entity modification
probe_entity_update() {
    log_json "info" "Testing UPDATE: Modify entity"

    local query="
        MATCH (e:Entity {uuid: '$TEST_ENTITY_UUID'})
        SET e.summary = 'Updated test entity for CRUD probe',
            e.updated_at = datetime()
        RETURN e.summary AS summary;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Entity UPDATE failed: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "Updated test entity"; then
        log_json "error" "Entity UPDATE did not apply: $response"
        return 1
    fi

    log_json "info" "Entity UPDATE successful"

    return 0
}

# Verify update (idempotency check)
probe_entity_verify_update() {
    log_json "info" "Testing IDEMPOTENCY: Verify update persisted"

    local query="
        MATCH (e:Entity {uuid: '$TEST_ENTITY_UUID'})
        RETURN e.summary AS summary;
    "

    local response
    response=$(cypher_query "$query")

    if ! echo "$response" | grep -q "Updated test entity"; then
        log_json "error" "Update did not persist: $response"
        return 1
    fi

    log_json "info" "Update verification successful - idempotent"

    return 0
}

# Test duplicate creation (idempotency)
probe_entity_idempotent_create() {
    log_json "info" "Testing IDEMPOTENCY: Attempt duplicate entity creation"

    # Try to create same entity again - should handle gracefully
    local query="
        MERGE (e:Entity {uuid: '$TEST_ENTITY_UUID'})
        ON CREATE SET e.name = 'ProbeTestEntity', e.created_at = datetime()
        ON MATCH SET e.updated_at = datetime()
        RETURN e.uuid AS uuid;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Idempotent CREATE failed: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "$TEST_ENTITY_UUID"; then
        log_json "error" "Idempotent CREATE returned unexpected: $response"
        return 1
    fi

    log_json "info" "Idempotent CREATE successful - MERGE works"

    return 0
}

# DELETE: Test entity deletion
probe_entity_delete() {
    log_json "info" "Testing DELETE: Remove entity"

    local query="
        MATCH (e:Entity {uuid: '$TEST_ENTITY_UUID'})
        DELETE e
        RETURN count(*) AS deleted_count;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Entity DELETE failed: $response"
        return 1
    fi

    log_json "info" "Entity DELETE successful"

    return 0
}

# Verify deletion
probe_entity_verify_deletion() {
    log_json "info" "Testing DELETE: Verify entity removed"

    local query="
        MATCH (e:Entity {uuid: '$TEST_ENTITY_UUID'})
        RETURN count(e) AS count;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Verification query failed: $response"
        return 1
    fi

    local count
    count=$(echo "$response" | grep -oE '[0-9]+' || echo "0")

    if [ "$count" -ne 0 ]; then
        log_json "error" "Entity still exists after deletion: count=$count"
        return 1
    fi

    log_json "info" "Entity deletion verified"

    return 0
}

# =============================================================================
# Probe Functions - Relationship Operations
# =============================================================================

# Create two entities and a relationship
probe_relationship_create() {
    log_json "info" "Testing CREATE: Create entities and relationship"

    local entity1_uuid="${TEST_ENTITY_UUID}_1"
    local entity2_uuid="${TEST_ENTITY_UUID}_2"

    local query="
        CREATE (e1:Entity {uuid: '$entity1_uuid', name: 'Entity1', created_at: datetime()})
        CREATE (e2:Entity {uuid: '$entity2_uuid', name: 'Entity2', created_at: datetime()})
        CREATE (e1)-[r:RELATED_TO {
            uuid: '$TEST_EDGE_UUID',
            fact: 'Entity1 is related to Entity2',
            created_at: datetime()
        }]->(e2)
        RETURN type(r) AS relation_type;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Relationship CREATE failed: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "RELATED_TO"; then
        log_json "error" "Relationship CREATE returned unexpected: $response"
        return 1
    fi

    log_json "info" "Relationship CREATE successful"

    return 0
}

# Query relationship
probe_relationship_read() {
    log_json "info" "Testing READ: Query relationship"

    local entity1_uuid="${TEST_ENTITY_UUID}_1"
    local entity2_uuid="${TEST_ENTITY_UUID}_2"

    local query="
        MATCH (e1:Entity {uuid: '$entity1_uuid'})-[r:RELATED_TO]->(e2:Entity {uuid: '$entity2_uuid'})
        RETURN r.fact AS fact, e1.name AS from_entity, e2.name AS to_entity;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Relationship READ failed: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "Entity1 is related to Entity2"; then
        log_json "error" "Relationship READ returned unexpected: $response"
        return 1
    fi

    log_json "info" "Relationship READ successful"

    return 0
}

# Update relationship
probe_relationship_update() {
    log_json "info" "Testing UPDATE: Modify relationship"

    local query="
        MATCH ()-[r:RELATED_TO {uuid: '$TEST_EDGE_UUID'}]->()
        SET r.fact = 'Updated: Entity1 is strongly related to Entity2',
            r.updated_at = datetime()
        RETURN r.fact AS fact;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Relationship UPDATE failed: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "Updated:"; then
        log_json "error" "Relationship UPDATE did not apply: $response"
        return 1
    fi

    log_json "info" "Relationship UPDATE successful"

    return 0
}

# Delete relationship (cascade with entities)
probe_relationship_delete() {
    log_json "info" "Testing DELETE: Remove entities and relationships"

    local entity1_uuid="${TEST_ENTITY_UUID}_1"
    local entity2_uuid="${TEST_ENTITY_UUID}_2"

    local query="
        MATCH (e1:Entity {uuid: '$entity1_uuid'})-[r]->(e2:Entity {uuid: '$entity2_uuid'})
        DETACH DELETE e1, e2
        RETURN count(*) AS deleted_count;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Relationship DELETE failed: $response"
        return 1
    fi

    log_json "info" "Relationship DELETE successful"

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting Graphiti entity/relationship CRUD probe"
    log_json "info" "Target URI: $NEO4J_URI"
    log_json "info" "Entity UUID: $TEST_ENTITY_UUID"

    # Check prerequisites
    check_cypher_shell

    # Validate environment
    if [ -z "$NEO4J_URI" ] || [ -z "$NEO4J_USER" ] || [ -z "$NEO4J_PASSWORD" ]; then
        log_json "error" "Required environment variables not set"
        exit 1
    fi

    # Cleanup any previous test data
    cleanup_test_data

    # Test Entity CRUD
    if ! probe_entity_create; then
        cleanup_test_data
        log_json "error" "Entity CREATE probe failed"
        exit 2
    fi

    if ! probe_entity_read; then
        cleanup_test_data
        log_json "error" "Entity READ probe failed"
        exit 3
    fi

    if ! probe_entity_update; then
        cleanup_test_data
        log_json "error" "Entity UPDATE probe failed"
        exit 4
    fi

    if ! probe_entity_verify_update; then
        cleanup_test_data
        log_json "error" "Entity UPDATE verification failed"
        exit 4
    fi

    if ! probe_entity_idempotent_create; then
        cleanup_test_data
        log_json "error" "Entity idempotent CREATE probe failed"
        exit 2
    fi

    if ! probe_entity_delete; then
        cleanup_test_data
        log_json "error" "Entity DELETE probe failed"
        exit 5
    fi

    if ! probe_entity_verify_deletion; then
        cleanup_test_data
        log_json "error" "Entity deletion verification failed"
        exit 5
    fi

    # Test Relationship Operations
    if ! probe_relationship_create; then
        cleanup_test_data
        log_json "error" "Relationship CREATE probe failed"
        exit 6
    fi

    if ! probe_relationship_read; then
        cleanup_test_data
        log_json "error" "Relationship READ probe failed"
        exit 6
    fi

    if ! probe_relationship_update; then
        cleanup_test_data
        log_json "error" "Relationship UPDATE probe failed"
        exit 6
    fi

    if ! probe_relationship_delete; then
        cleanup_test_data
        log_json "error" "Relationship DELETE probe failed"
        exit 6
    fi

    # Final cleanup
    cleanup_test_data

    # All probes passed
    log_json "info" "All Graphiti entity/relationship CRUD probes passed successfully"
    exit 0
}

# Run main function
main "$@"

#!/bin/bash
##############################################################################
# Graphiti Graph Operations Probe Script
#
# Purpose: Verify Graphiti graph operations (nodes, edges, episodes)
# Compliance: Law of Runtime Truth - verify graph operations work
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
#   2 - Node operations failed
#   3 - Edge operations failed
#   4 - Episode operations failed
#   5 - cypher-shell not available
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
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# Test data prefix
TEST_PREFIX="probe_test_$(date +%s)_"

# =============================================================================
# Utility Functions
# =============================================================================

log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_graph.sh\"}"
}

check_cypher_shell() {
    if ! command -v cypher-shell &> /dev/null; then
        log_json "error" "cypher-shell is not installed or not in PATH"
        exit 5
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

    local cleanup_query="
        MATCH (e:Entity {name: '${TEST_PREFIX}TestEntity'})
        DETACH DELETE e;

        MATCH (e:Episodic {name: '${TEST_PREFIX}TestEpisode'})
        DETACH DELETE e;
    "

    cypher_query "$cleanup_query" > /dev/null 2>&1 || true
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Entity Node Creation
probe_create_entity() {
    log_json "info" "Testing entity node creation"

    local entity_name="${TEST_PREFIX}TestEntity"
    local query="
        CREATE (e:Entity {
            uuid: '${entity_name}_uuid',
            name: '$entity_name',
            created_at: datetime(),
            summary: 'Test entity for probe'
        })
        RETURN e.name AS name;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Failed to create entity: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "$entity_name"; then
        log_json "error" "Entity creation returned unexpected result: $response"
        return 1
    fi

    log_json "info" "Entity node created successfully"

    return 0
}

# Probe 2: Entity Node Query
probe_query_entity() {
    log_json "info" "Testing entity node query"

    local entity_name="${TEST_PREFIX}TestEntity"
    local query="
        MATCH (e:Entity {name: '$entity_name'})
        RETURN e.name AS name, e.summary AS summary;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Failed to query entity: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "$entity_name"; then
        log_json "error" "Entity query returned unexpected result: $response"
        return 1
    fi

    log_json "info" "Entity node queried successfully"

    return 0
}

# Probe 3: Entity Edge Creation
probe_create_edge() {
    log_json "info" "Testing entity edge creation"

    local entity_name="${TEST_PREFIX}TestEntity"
    local query="
        MATCH (e:Entity {name: '$entity_name'})
        CREATE (e)-[r:TEST_RELATIONSHIP {
            uuid: '${entity_name}_edge_uuid',
            fact: 'Test edge for probe',
            created_at: datetime()
        }]->(e)
        RETURN type(r) AS relation_type;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Failed to create edge: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "TEST_RELATIONSHIP"; then
        log_json "error" "Edge creation returned unexpected result: $response"
        return 1
    fi

    log_json "info" "Entity edge created successfully"

    return 0
}

# Probe 4: Episode Node Creation
probe_create_episode() {
    log_json "info" "Testing episode node creation"

    local episode_name="${TEST_PREFIX}TestEpisode"
    local query="
        CREATE (ep:Episodic {
            uuid: '${episode_name}_uuid',
            name: '$episode_name',
            episode_body: 'Test episode content for probe',
            source: 'text',
            created_at: datetime(),
            valid_at: datetime()
        })
        RETURN ep.name AS name;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Failed to create episode: $response"
        return 1
    fi

    if ! echo "$response" | grep -q "$episode_name"; then
        log_json "error" "Episode creation returned unexpected result: $response"
        return 1
    fi

    log_json "info" "Episode node created successfully"

    return 0
}

# Probe 5: Temporal Query
probe_temporal_query() {
    log_json "info" "Testing temporal episode query"

    local query="
        MATCH (ep:Episodic)
        WHERE ep.valid_at <= datetime()
        RETURN count(ep) AS count;
    "

    local response
    response=$(cypher_query "$query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Failed temporal query: $response"
        return 1
    fi

    log_json "info" "Temporal query successful"

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting Graphiti graph operations probe"
    log_json "info" "Target URI: $NEO4J_URI"
    log_json "info" "Test prefix: $TEST_PREFIX"

    # Check prerequisites
    check_cypher_shell

    # Validate environment
    if [ -z "$NEO4J_URI" ] || [ -z "$NEO4J_USER" ] || [ -z "$NEO4J_PASSWORD" ]; then
        log_json "error" "Required environment variables not set"
        exit 1
    fi

    # Cleanup any previous test data
    cleanup_test_data

    # Run probes
    if ! probe_create_entity; then
        cleanup_test_data
        log_json "error" "Entity creation probe failed"
        exit 2
    fi

    if ! probe_query_entity; then
        cleanup_test_data
        log_json "error" "Entity query probe failed"
        exit 2
    fi

    if ! probe_create_edge; then
        cleanup_test_data
        log_json "error" "Edge creation probe failed"
        exit 3
    fi

    if ! probe_create_episode; then
        cleanup_test_data
        log_json "error" "Episode creation probe failed"
        exit 4
    fi

    if ! probe_temporal_query; then
        cleanup_test_data
        log_json "error" "Temporal query probe failed"
        exit 4
    fi

    # Cleanup test data
    cleanup_test_data

    # All probes passed
    log_json "info" "All Graphiti graph operation probes passed successfully"
    exit 0
}

# Run main function
main "$@"

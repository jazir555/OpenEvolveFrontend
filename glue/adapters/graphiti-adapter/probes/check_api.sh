#!/bin/bash
##############################################################################
# Graphiti API Probe Script
#
# Purpose: Verify Graphiti/Neo4j API endpoints are accessible and responding
# Compliance: Law of Runtime Truth - verify before implementation
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
#   2 - Neo4j connection failed
#   3 - Index creation failed
#   4 - Query test failed
#   5 - cypher-shell not available
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration (from environment variables)
# =============================================================================

NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
TIMEOUT_MS="${TIMEOUT_MS:-10000}"
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# =============================================================================
# Utility Functions
# =============================================================================

# Log JSON Lines output
log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_api.sh\"}"
}

# Check if cypher-shell is available
check_cypher_shell() {
    if ! command -v cypher-shell &> /dev/null; then
        log_json "error" "cypher-shell is not installed or not in PATH"
        exit 5
    fi
}

# Execute Cypher query with timeout
cypher_query() {
    local query="$1"

    echo "$query" | cypher-shell \
        -a "$NEO4J_URI" \
        -u "$NEO4J_USER" \
        -p "$NEO4J_PASSWORD" \
        --format plain \
        2>&1 || echo "ERROR: Query failed"
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Neo4j Connection Check
probe_connection() {
    log_json "info" "Testing Neo4j connection: $NEO4J_URI"

    local response
    response=$(cypher_query "RETURN 1 AS test;")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Neo4j connection failed: $response"
        return 1
    fi

    # Check if we got the expected result
    if ! echo "$response" | grep -q "1"; then
        log_json "error" "Neo4j returned unexpected response: $response"
        return 1
    fi

    log_json "info" "Neo4j connection successful"

    return 0
}

# Probe 2: Graphiti Indices Check
probe_indices() {
    log_json "info" "Testing Graphiti indices"

    # Check for Graphiti-specific indices
    local response
    response=$(cypher_query "SHOW INDEXES;")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Failed to query indices: $response"
        return 1
    fi

    # Count indices
    local index_count
    index_count=$(echo "$response" | wc -l)

    log_json "info" "Found $index_count indices in database"

    # Try to create Graphiti indices if they don't exist
    log_json "info" "Attempting to ensure Graphiti indices exist"

    local create_indices_query="
        CREATE INDEX entity_node_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
        CREATE INDEX episodic_node_created_at_index IF NOT EXISTS FOR (e:Episodic) ON (e.created_at);
        CREATE INDEX entity_edge_created_at_index IF NOT EXISTS FOR (e:EntityEdge) ON (e.created_at);
    "

    response=$(cypher_query "$create_indices_query")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Failed to create indices: $response"
        return 1
    fi

    log_json "info" "Graphiti indices verified/created"

    return 0
}

# Probe 3: Basic Graph Query Test
probe_graph_query() {
    log_json "info" "Testing basic graph query"

    # Query for entity nodes (should exist or be empty)
    local response
    response=$(cypher_query "MATCH (e:Entity) RETURN count(e) AS entity_count;")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Graph query failed: $response"
        return 1
    fi

    # Extract count
    local count
    count=$(echo "$response" | grep -E '[0-9]+' || echo "0")

    log_json "info" "Graph query successful - entity count: $count"

    return 0
}

# Probe 4: Temporal Query Test
probe_temporal_query() {
    log_json "info" "Testing temporal query capabilities"

    # Query episodes with time filtering
    local response
    response=$(cypher_query "
        MATCH (ep:Episodic)
        WHERE ep.valid_at <= datetime()
        RETURN count(ep) AS valid_episode_count
        LIMIT 1;
    ")

    if echo "$response" | grep -q "ERROR"; then
        log_json "error" "Temporal query failed: $response"
        return 1
    fi

    log_json "info" "Temporal query successful"

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting Graphiti API probe"
    log_json "info" "Target URI: $NEO4J_URI"
    log_json "info" "User: $NEO4J_USER"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    # Check prerequisites
    check_cypher_shell

    # Validate environment
    if [ -z "$NEO4J_URI" ]; then
        log_json "error" "NEO4J_URI environment variable is not set"
        exit 1
    fi

    if [ -z "$NEO4J_USER" ]; then
        log_json "error" "NEO4J_USER environment variable is not set"
        exit 1
    fi

    if [ -z "$NEO4J_PASSWORD" ]; then
        log_json "error" "NEO4J_PASSWORD environment variable is not set"
        exit 1
    fi

    # Run probes sequentially (fail fast on first error)
    if ! probe_connection; then
        log_json "error" "Connection probe failed"
        exit 2
    fi

    if ! probe_indices; then
        log_json "error" "Indices probe failed"
        exit 3
    fi

    if ! probe_graph_query; then
        log_json "error" "Graph query probe failed"
        exit 4
    fi

    if ! probe_temporal_query; then
        log_json "error" "Temporal query probe failed"
        exit 4
    fi

    # All probes passed
    log_json "info" "All Graphiti API probes passed successfully"
    exit 0
}

# Run main function
main "$@"

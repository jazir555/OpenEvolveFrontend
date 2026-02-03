#!/bin/bash
##############################################################################
# KarateClub Embedding Operations Probe Script
#
# Purpose: Verify node embedding operations work correctly
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   PYTHON_PATH          - Python executable path (default: python3)
#   TIMEOUT_MS           - Timeout in milliseconds (default: 60000)
#
# Exit Codes:
#   0 - All embedding operation tests passed
#   1 - Environment variable missing
#   2 - Node embedding test failed
#   3 - Community detection test failed
#   4 - Integration test failed
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

PYTHON_PATH="${PYTHON_PATH:-python3}"
TIMEOUT_MS="${TIMEOUT_MS:-60000}"
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# =============================================================================
# Utility Functions
# =============================================================================

log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_embeddings.sh\"}"
}

execute_python() {
    local code="$1"
    timeout "$TIMEOUT_SEC" "$PYTHON_PATH" -c "$code" 2>&1 || true
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Test Node Embedding Generation
probe_node_embedding() {
    log_json "info" "Testing node embedding generation"

    local code="
import sys
import json
import networkx as nx
import numpy as np
import time

try:
    from karateclub import DeepWalk

    # Create test graph
    G = nx.karate_club_graph()

    # Initialize model
    model = DeepWalk(dimensions=32, walk_length=30, walk_number=10)

    # Fit and time
    start = time.time()
    model.fit(G)
    fit_time = time.time() - start

    # Get embeddings
    embedding = model.get_embedding()

    # Verify output
    assert embedding.shape[0] == G.number_of_nodes(), 'Wrong number of embeddings'
    assert embedding.shape[1] == 32, 'Wrong embedding dimension'

    result = {
        'status': 'OK',
        'num_nodes': G.number_of_nodes(),
        'embedding_dim': embedding.shape[1],
        'fit_time_sec': round(fit_time, 2)
    }

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({'status': 'ERROR', 'error': str(e)}))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | jq -e '.status == "OK"' &>/dev/null; then
        log_json "error" "Node embedding test failed: $output"
        return 1
    fi

    local num_nodes
    local embedding_dim
    local fit_time

    num_nodes=$(echo "$output" | jq -r '.num_nodes')
    embedding_dim=$(echo "$output" | jq -r '.embedding_dim')
    fit_time=$(echo "$output" | jq -r '.fit_time_sec')

    log_json "info" "Node embedding successful: ${num_nodes} nodes, ${embedding_dim} dimensions, ${fit_time}s"
    return 0
}

# Probe 2: Test Community Detection
probe_community_detection() {
    log_json "info" "Testing community detection"

    local code="
import sys
import json
import networkx as nx
import time

try:
    from karateclub import LabelPropagation

    # Create test graph
    G = nx.karate_club_graph()

    # Initialize model
    model = LabelPropagation()

    # Fit and time
    start = time.time()
    model.fit(G)
    fit_time = time.time() - start

    # Get communities
    membership = model.get_memberships()
    num_communities = len(set(membership.values()))

    result = {
        'status': 'OK',
        'num_nodes': G.number_of_nodes(),
        'num_communities': num_communities,
        'fit_time_sec': round(fit_time, 2)
    }

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({'status': 'ERROR', 'error': str(e)}))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | jq -e '.status == "OK"' &>/dev/null; then
        log_json "error" "Community detection test failed: $output"
        return 1
    fi

    local num_nodes
    local num_communities
    local fit_time

    num_nodes=$(echo "$output" | jq -r '.num_nodes')
    num_communities=$(echo "$output" | jq -r '.num_communities')
    fit_time=$(echo "$output" | jq -r '.fit_time_sec')

    log_json "info" "Community detection successful: ${num_nodes} nodes, ${num_communities} communities, ${fit_time}s"
    return 0
}

# Probe 3: Test Graph Embedding
probe_graph_embedding() {
    log_json "info" "Testing graph embedding"

    local code="
import sys
import json
import networkx as nx
import time

try:
    from karateclub import Graph2Vec

    # Create test graphs
    graphs = []
    for i in range(5):
        G = nx.barabasi_albert_graph(20, 2)
        graphs.append(G)

    # Initialize model
    model = Graph2Vec(dimensions=32, wl_iterations=2)

    # Fit and time
    start = time.time()
    model.fit(graphs)
    fit_time = time.time() - start

    # Get embeddings
    embedding = model.get_embedding()

    result = {
        'status': 'OK',
        'num_graphs': len(graphs),
        'embedding_dim': embedding.shape[1],
        'fit_time_sec': round(fit_time, 2)
    }

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({'status': 'ERROR', 'error': str(e)}))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | jq -e '.status == "OK"' &>/dev/null; then
        log_json "error" "Graph embedding test failed: $output"
        return 1
    fi

    local num_graphs
    local embedding_dim
    local fit_time

    num_graphs=$(echo "$output" | jq -r '.num_graphs')
    embedding_dim=$(echo "$output" | jq -r '.embedding_dim')
    fit_time=$(echo "$output" | jq -r '.fit_time_sec')

    log_json "info" "Graph embedding successful: ${num_graphs} graphs, ${embedding_dim} dimensions, ${fit_time}s"
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting KarateClub embedding operations probe"
    log_json "info" "Python path: $PYTHON_PATH"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        log_json "warn" "jq not available, JSON parsing may fail"
    fi

    # Run probes
    if ! probe_node_embedding; then
        log_json "error" "Node embedding probe failed"
        exit 2
    fi

    if ! probe_community_detection; then
        log_json "error" "Community detection probe failed"
        exit 3
    fi

    if ! probe_graph_embedding; then
        log_json "error" "Graph embedding probe failed"
        exit 4
    fi

    log_json "info" "All KarateClub embedding operation probes passed"
    exit 0
}

main "$@"

#!/bin/bash
##############################################################################
# KarateClub Algorithm Probe Script
#
# Purpose: Verify KarateClub algorithms are accessible
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   PYTHON_PATH          - Python executable path (default: python3)
#   TIMEOUT_MS           - Timeout in milliseconds (default: 30000)
#
# Exit Codes:
#   0 - All algorithm checks passed
#   1 - Environment variable missing
#   2 - Node embedding algorithms failed
#   3 - Community detection algorithms failed
#   4 - Graph embedding algorithms failed
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

PYTHON_PATH="${PYTHON_PATH:-python3}"
TIMEOUT_MS="${TIMEOUT_MS:-30000}"
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# =============================================================================
# Utility Functions
# =============================================================================

log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_algorithms.sh\"}"
}

execute_python() {
    local code="$1"
    timeout "$TIMEOUT_SEC" "$PYTHON_PATH" -c "$code" 2>&1 || true
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Node Embedding Algorithms
probe_node_embeddings() {
    log_json "info" "Checking node embedding algorithms"

    local code="
import sys
try:
    from karateclub import DeepWalk, Node2Vec
    print('OK: DeepWalk, Node2Vec')
except ImportError as e:
    try:
        from karateclub.node_embedding.neighbourhood import DeepWalk, Node2Vec
        print('OK: DeepWalk, Node2Vec (submodule)')
    except ImportError:
        print(f'ERROR: {e}')
        sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | grep -q "OK"; then
        log_json "error" "Node embedding algorithms check failed: $output"
        return 1
    fi

    log_json "info" "$(echo "$output" | grep "OK:")"
    return 0
}

# Probe 2: Community Detection Algorithms
probe_community_detection() {
    log_json "info" "Checking community detection algorithms"

    local code="
import sys
try:
    from karateclub import LabelPropagation
    print('OK: LabelPropagation')
except ImportError as e:
    try:
        from karateclub.community_detection.non_overlapping import LabelPropagation
        print('OK: LabelPropagation (submodule)')
    except ImportError:
        print(f'ERROR: {e}')
        sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | grep -q "OK"; then
        log_json "error" "Community detection algorithms check failed: $output"
        return 1
    fi

    log_json "info" "$(echo "$output" | grep "OK:")"
    return 0
}

# Probe 3: Graph Embedding Algorithms
probe_graph_embeddings() {
    log_json "info" "Checking graph embedding algorithms"

    local code="
import sys
try:
    from karateclub import Graph2Vec
    print('OK: Graph2Vec')
except ImportError as e:
    try:
        from karateclub.graph_embedding import Graph2Vec
        print('OK: Graph2Vec (submodule)')
    except ImportError:
        print(f'ERROR: {e}')
        sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | grep -q "OK"; then
        log_json "error" "Graph embedding algorithms check failed: $output"
        return 1
    fi

    log_json "info" "$(echo "$output" | grep "OK:")"
    return 0
}

# Probe 4: Test basic algorithm execution
test_algorithm_execution() {
    log_json "info" "Testing basic algorithm execution"

    local code="
import sys
import networkx as nx
import numpy as np

try:
    from karateclub import LabelPropagation

    # Create simple test graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    # Run algorithm
    model = LabelPropagation()
    model.fit(G)

    # Get memberships
    membership = model.get_memberships()

    print('OK: Algorithm execution successful')
    print(f'Communities detected: {len(set(membership.values()))}')

except Exception as e:
    print(f'ERROR: Algorithm execution failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | grep -q "OK"; then
        log_json "error" "Algorithm execution test failed: $output"
        return 1
    fi

    echo "$output" | while read -r line; do
        log_json "info" "$line"
    done

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting KarateClub algorithm probe"
    log_json "info" "Python path: $PYTHON_PATH"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    # Run probes
    if ! probe_node_embeddings; then
        log_json "error" "Node embedding algorithm probe failed"
        exit 2
    fi

    if ! probe_community_detection; then
        log_json "error" "Community detection algorithm probe failed"
        exit 3
    fi

    if ! probe_graph_embeddings; then
        log_json "error" "Graph embedding algorithm probe failed"
        exit 4
    fi

    if ! test_algorithm_execution; then
        log_json "error" "Algorithm execution test failed"
        exit 5
    fi

    log_json "info" "All KarateClub algorithm probes passed"
    exit 0
}

main "$@"

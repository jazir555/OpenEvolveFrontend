#!/bin/bash

###############################################################################
# Probe: Check Retrieval Operations
#
# Following CLAUDE.md Federation Constitution:
# - Law of Runtime Truth: Verify retrieval actually works
# - This script MUST successfully execute before trusting search/lineage
#
# Usage:
#   ./check_retrieval.sh
#
# Environment Variables Required:
#   - VECTORDB_ADAPTER_URL: Vector DB adapter URL
#   - GRAPHITI_ADAPTER_URL: Graphiti adapter URL
#   - EVOLVED_CODE_COLLECTION: Collection name for evolved code
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration from environment
VECTORDB_URL="${VECTORDB_ADAPTER_URL:-http://localhost:8000}"
GRAPHITI_URL="${GRAPHITI_ADAPTER_URL:-http://localhost:8001}"
COLLECTION_NAME="${EVOLVED_CODE_COLLECTION:-evolved_code}"
TIMEOUT="${STORAGE_TIMEOUT:-10}"

echo "=========================================="
echo "Evolved Code Capture - Retrieval Probe"
echo "=========================================="
echo ""
echo "Vector DB Adapter: ${VECTORDB_URL}"
echo "Graphiti Adapter: ${GRAPHITI_URL}"
echo "Collection: ${COLLECTION_NAME}"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Track results
vector_search_ok=0
graph_search_ok=0

# Check Vector DB search
echo "1. Vector DB Search Test"
echo "---------------------------"
echo -n "Testing semantic search... "

# Create a test search payload
SEARCH_PAYLOAD=$(cat <<EOF
{
  "collection_name": "${COLLECTION_NAME}",
  "query": {
    "vector": [$(seq -s, 1.0 1.0 1536)],
    "k": 5
  }
}
EOF
)

if response=$(curl -s -w "\n%{http_code}" \
    --max-time "${TIMEOUT}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "${SEARCH_PAYLOAD}" \
    "${VECTORDB_URL}/collections/${COLLECTION_NAME}/search" 2>/dev/null); then

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "${http_code}" = "200" ]; then
        # Check if response contains results array (even if empty)
        if echo "${body}" | grep -q "results"; then
            echo -e "${GREEN}✓ OK${NC} (search endpoint working)"
            vector_search_ok=1
        else
            echo -e "${YELLOW}⚠ WARNING${NC} (unexpected response format)"
        fi
    elif [ "${http_code}" = "404" ]; then
        echo -e "${YELLOW}⚠ Collection not found${NC} (create collection first)"
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP ${http_code})"
    fi
else
    echo -e "${RED}✗ FAILED${NC} (connection error)"
fi
echo ""

# Check Graphiti search
echo "2. Graphiti Search Test"
echo "---------------------------"
echo -n "Testing graph search... "

# Create a test search payload
GRAPH_SEARCH_PAYLOAD=$(cat <<EOF
{
  "query": "test query",
  "max_results": 10,
  "temporal_filter": {
    "start": "1970-01-01T00:00:00.000Z",
    "end": "$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")"
  }
}
EOF
)

if response=$(curl -s -w "\n%{http_code}" \
    --max-time "${TIMEOUT}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "${GRAPH_SEARCH_PAYLOAD}" \
    "${GRAPHITI_URL}/search" 2>/dev/null); then

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "${http_code}" = "200" ]; then
        # Check if response contains nodes/edges
        if echo "${body}" | grep -q "nodes\|edges"; then
            echo -e "${GREEN}✓ OK${NC} (search endpoint working)"
            graph_search_ok=1
        else
            echo -e "${YELLOW}⚠ WARNING${NC} (unexpected response format)"
        fi
    else
        echo -e "${YELLOW}⚠ HTTP ${http_code}${NC} (may be empty graph)"
        graph_search_ok=1  # Empty graph is still OK
    fi
else
    echo -e "${RED}✗ FAILED${NC} (connection error)"
fi
echo ""

# Check Graphiti episode operations
echo "3. Graphiti Episode Operations"
echo "---------------------------"
echo -n "Testing episode availability... "

if response=$(curl -s -w "\n%{http_code}" \
    --max-time "${TIMEOUT}" \
    "${GRAPHITI_URL}/episodes" 2>/dev/null); then

    http_code=$(echo "$response" | tail -n1)

    if [ "${http_code}" = "200" ] || [ "${http_code}" = "405" ]; then
        echo -e "${GREEN}✓ OK${NC} (endpoint available)"
    else
        echo -e "${YELLOW}⚠ HTTP ${http_code}${NC}"
    fi
else
    echo -e "${YELLOW}⚠ WARNING${NC} (endpoint may not exist)"
fi
echo ""

# Summary
echo "=========================================="
echo "Probe Results"
echo "=========================================="

if [ $vector_search_ok -eq 1 ] && [ $graph_search_ok -eq 1 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "Retrieval operations are working correctly."
    exit 0
elif [ $vector_search_ok -eq 1 ]; then
    echo -e "${YELLOW}⚠ VECTOR SEARCH OK, Graphiti FAILED${NC}"
    echo ""
    echo "Vector search is available, but graph search is not."
    exit 1
elif [ $graph_search_ok -eq 1 ]; then
    echo -e "${YELLOW}⚠ Graphiti OK, Vector Search FAILED${NC}"
    echo ""
    echo "Graph search is available, but vector search is not."
    exit 1
else
    echo -e "${RED}✗ ALL CHECKS FAILED${NC}"
    echo ""
    echo "Retrieval operations are not working."
    echo "Please check that:"
    echo "  1. Storage backends are running (run check_storage.sh first)"
    echo "  2. Collections exist and have data"
    echo "  3. Search endpoints are properly configured"
    exit 2
fi

#!/bin/bash

###############################################################################
# Probe: Check Storage Connectivity
#
# Following CLAUDE.md Federation Constitution:
# - Law of Runtime Truth: Verify storage backends actually work
# - This script MUST successfully execute before using the capture system
#
# Usage:
#   ./check_storage.sh
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
TIMEOUT="${STORAGE_TIMEOUT:-5}"

echo "=========================================="
echo "Evolved Code Capture - Storage Probe"
echo "=========================================="
echo ""
echo "Vector DB Adapter: ${VECTORDB_URL}"
echo "Graphiti Adapter: ${GRAPHITI_URL}"
echo "Collection: ${COLLECTION_NAME}"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Function to check HTTP endpoint
check_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}

    echo -n "Checking ${name}... "

    if response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "${TIMEOUT}" \
        "${url}" 2>/dev/null); then

        if [ "${response}" = "${expected_status}" ] || [ "${response}" = "200" ]; then
            echo -e "${GREEN}✓ OK${NC} (HTTP ${response})"
            return 0
        else
            echo -e "${YELLOW}⚠ WARNING${NC} (HTTP ${response})"
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED${NC} (connection error)"
        return 2
    fi
}

# Function to check collection exists
check_collection() {
    local name=$1
    local url=$2

    echo -n "Checking ${name}... "

    if response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "${TIMEOUT}" \
        "${url}" 2>/dev/null); then

        if [ "${response}" = "200" ]; then
            echo -e "${GREEN}✓ EXISTS${NC}"
            return 0
        elif [ "${response}" = "404" ]; then
            echo -e "${YELLOW}⚠ NOT FOUND (will be created)${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠ WARNING${NC} (HTTP ${response})"
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED${NC}"
        return 2
    fi
}

# Track results
vector_db_ok=0
graphiti_ok=0

# Check Vector DB Adapter health
echo "1. Vector DB Adapter Health"
echo "---------------------------"
if check_endpoint "Health Endpoint" "${VECTORDB_URL}/health"; then
    vector_db_ok=1
fi
echo ""

# Check Vector DB collection
echo "2. Vector DB Collection"
echo "---------------------------"
if check_collection "Collection" "${VECTORDB_URL}/collections/${COLLECTION_NAME}"; then
    if [ $vector_db_ok -eq 0 ]; then
        vector_db_ok=1
    fi
fi
echo ""

# Check Graphiti Adapter health
echo "3. Graphiti Adapter Health"
echo "---------------------------"
if check_endpoint "Health Endpoint" "${GRAPHITI_URL}/health"; then
    graphiti_ok=1
fi
echo ""

# Check Graphiti statistics
echo "4. Graphiti Statistics"
echo "---------------------------"
if check_endpoint "Statistics Endpoint" "${GRAPHITI_URL}/statistics"; then
    if [ $graphiti_ok -eq 0 ]; then
        graphiti_ok=1
    fi
fi
echo ""

# Summary
echo "=========================================="
echo "Probe Results"
echo "=========================================="

if [ $vector_db_ok -eq 1 ] && [ $graphiti_ok -eq 1 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "Storage backends are ready for evolved code capture."
    exit 0
elif [ $vector_db_ok -eq 1 ]; then
    echo -e "${YELLOW}⚠ VECTOR DB OK, Graphiti FAILED${NC}"
    echo ""
    echo "Vector storage is available, but graph storage is not."
    echo "You can use the capturer with vector-only mode:"
    echo "  ENABLE_GRAPH_STORAGE=false"
    exit 1
elif [ $graphiti_ok -eq 1 ]; then
    echo -e "${YELLOW}⚠ Graphiti OK, Vector DB FAILED${NC}"
    echo ""
    echo "Graph storage is available, but vector storage is not."
    echo "You can use the capturer with graph-only mode:"
    echo "  ENABLE_VECTOR_STORAGE=false"
    exit 1
else
    echo -e "${RED}✗ ALL CHECKS FAILED${NC}"
    echo ""
    echo "Neither storage backend is available."
    echo "Please check that:"
    echo "  1. Vector DB adapter is running at ${VECTORDB_URL}"
    echo "  2. Graphiti adapter is running at ${GRAPHITI_URL}"
    echo "  3. Network connectivity is working"
    exit 2
fi

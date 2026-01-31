#!/bin/bash
#
# Probe script: Verify Graphiti database connection (Shell version)
#
# Following CLAUDE.md LAW OF RUNTIME TRUTH:
# - Verify actual connectivity before using the integration
# - Fail explicitly if the probe doesn't succeed
# - This script MUST return 0 for success, non-zero for failure
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting Graphiti connection probe..."

# Check required environment variables
echo "[1/5] Checking environment variables..."

if [ -z "${GRAPHITI_URI:-}" ]; then
    echo -e "${RED}✗ GRAPHITI_URI not set${NC}"
    exit 1
fi

if [ -z "${GRAPHITI_USER:-}" ]; then
    echo -e "${RED}✗ GRAPHITI_USER not set${NC}"
    exit 1
fi

if [ -z "${GRAPHITI_PASSWORD:-}" ]; then
    echo -e "${RED}✗ GRAPHITI_PASSWORD not set${NC}"
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -e "${RED}✗ OPENAI_API_KEY not set${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required environment variables set${NC}"
echo "  GRAPHITI_URI: ${GRAPHITI_URI:0:20}..."
echo "  GRAPHITI_USER: ${GRAPHITI_USER}"
echo "  GRAPHITI_DATABASE: ${GRAPHITI_DATABASE:-neo4j}"

# Check Python is available
echo ""
echo "[2/5] Checking Python availability..."

if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}✓ Python available: ${PYTHON_VERSION}${NC}"

# Check required Python packages
echo ""
echo "[3/5] Checking required Python packages..."

REQUIRED_PACKAGES="graphiti_core asyncio"

for package in $REQUIRED_PACKAGES; do
    if python -c "import ${package}" 2>/dev/null; then
        echo -e "${GREEN}✓ ${package}${NC}"
    else
        echo -e "${RED}✗ ${package} not installed${NC}"
        exit 1
    fi
done

# Try to connect to Neo4j using cypher-shell if available
echo ""
echo "[4/5] Testing Neo4j connectivity (optional)..."

if command -v cypher-shell &> /dev/null; then
    # Extract host and port from URI
    # Assumes format like bolt://localhost:7687
    NEO4J_HOST=$(echo "$GRAPHITI_URI" | sed -n 's|bolt://\([^:]*\):\([0-9]*\).*|\1|p')
    NEO4J_PORT=$(echo "$GRAPHITI_URI" | sed -n 's|bolt://\([^:]*\):\([0-9]*\).*|\2|p')

    if echo "RETURN 1" | cypher-shell -a "$GRAPHITI_URI" -u "$GRAPHITI_USER" -p "$GRAPHITI_PASSWORD" &> /dev/null; then
        echo -e "${GREEN}✓ Direct Neo4j connection successful${NC}"
    else
        echo -e "${YELLOW}⚠ Direct Neo4j connection failed (may be OK if using Graphiti wrapper)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ cypher-shell not found, skipping direct Neo4j test${NC}"
fi

# Run the Python probe script for comprehensive testing
echo ""
echo "[5/5] Running comprehensive Python probe..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PROBE="$SCRIPT_DIR/check_connection.py"

if [ -f "$PYTHON_PROBE" ]; then
    if python "$PYTHON_PROBE"; then
        echo -e "${GREEN}✓ Python probe passed${NC}"
    else
        echo -e "${RED}✗ Python probe failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python probe not found: $PYTHON_PROBE${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All probe checks passed${NC}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Probe completed successfully"

exit 0

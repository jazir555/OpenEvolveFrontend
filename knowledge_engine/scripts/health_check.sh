#!/bin/bash

# Neo4j Health Check Script
# OpenEvolve Knowledge Engine - Phase 1.1.1
#
# This script performs comprehensive health checks on the Neo4j instance
# Usage: ./health_check.sh

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration from environment variables
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-openevolve2026}"
NEO4J_HTTP_URI="${NEO4J_HTTP_URI:-http://localhost:7474}"

echo "================================================"
echo "Neo4j Health Check - OpenEvolve Knowledge Engine"
echo "================================================"
echo ""

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "WARNING" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# CHECK 1: Required Commands
# ============================================================================
echo "1. Checking required commands..."

REQUIRED_COMMANDS=("curl" "cypher-shell" "grep" "awk")
ALL_COMMANDS_OK=true

for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if command_exists "$cmd"; then
        print_status "OK" "$cmd is available"
    else
        print_status "FAIL" "$cmd is not available"
        ALL_COMMANDS_OK=false
    fi
done

if [ "$ALL_COMMANDS_OK" = false ]; then
    echo -e "${RED}ERROR: Required commands are missing. Please install them.${NC}"
    exit 1
fi

echo ""

# ============================================================================
# CHECK 2: Environment Variables
# ============================================================================
echo "2. Checking environment variables..."

ENV_VARS_OK=true

if [ -z "$NEO4J_URI" ]; then
    print_status "FAIL" "NEO4J_URI is not set"
    ENV_VARS_OK=false
else
    print_status "OK" "NEO4J_URI=$NEO4J_URI"
fi

if [ -z "$NEO4J_USER" ]; then
    print_status "FAIL" "NEO4J_USER is not set"
    ENV_VARS_OK=false
else
    print_status "OK" "NEO4J_USER=$NEO4J_USER"
fi

if [ -z "$NEO4J_PASSWORD" ]; then
    print_status "FAIL" "NEO4J_PASSWORD is not set"
    ENV_VARS_OK=false
else
    print_status "OK" "NEO4J_PASSWORD is set (hidden)"
fi

echo ""

# ============================================================================
# CHECK 3: HTTP Endpoint Availability
# ============================================================================
echo "3. Checking HTTP endpoint availability..."

HTTP_OK=true
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$NEO4J_HTTP_URI" || echo "000")

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
    print_status "OK" "HTTP endpoint is reachable (HTTP $HTTP_CODE)"
else
    print_status "FAIL" "HTTP endpoint is not reachable (HTTP $HTTP_CODE)"
    HTTP_OK=false
fi

echo ""

# ============================================================================
# CHECK 4: Bolt Protocol Connection
# ============================================================================
echo "4. Checking Bolt protocol connection..."

BOLT_OK=true
if echo "RETURN 1" | cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" >/dev/null 2>&1; then
    print_status "OK" "Bolt protocol connection successful"
else
    print_status "FAIL" "Bolt protocol connection failed"
    BOLT_OK=false
fi

echo ""

# ============================================================================
# CHECK 5: Database Version
# ============================================================================
echo "5. Checking database version..."

if [ "$BOLT_OK" = true ]; then
    VERSION=$(echo "CALL dbms.components() YIELD versions RETURN versions[0] as version" | \
        cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" | \
        grep -E "^[0-9]" | tr -d ' ')

    if [ -n "$VERSION" ]; then
        print_status "OK" "Neo4j version: $VERSION"

        # Check if version is 5.26+
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)

        if [ "$MAJOR" -gt 5 ] || ([ "$MAJOR" -eq 5 ] && [ "$MINOR" -ge 26 ]); then
            print_status "OK" "Version 5.26+ requirement satisfied"
        else
            print_status "WARNING" "Version is below 5.26 (current: $VERSION)"
        fi
    else
        print_status "WARNING" "Could not determine database version"
    fi
else
    print_status "SKIP" "Cannot check version (Bolt connection failed)"
fi

echo ""

# ============================================================================
# CHECK 6: Database Size
# ============================================================================
echo "6. Checking database size..."

if [ "$BOLT_OK" = true ]; then
    STATS=$(echo "CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Primitive count') YIELD attributes RETURN attributes.NodeCount as nodeCount" | \
        cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" | \
        grep -E "^[0-9]" | tr -d ' ')

    if [ -n "$STATS" ]; then
        print_status "OK" "Database contains $STATS nodes"

        # Check if database is empty
        if [ "$STATS" -eq 0 ]; then
            print_status "WARNING" "Database is empty. Run init_neo4j.cypher to initialize."
        fi
    else
        print_status "WARNING" "Could not retrieve node count"
    fi
else
    print_status "SKIP" "Cannot check database size (Bolt connection failed)"
fi

echo ""

# ============================================================================
# CHECK 7: Constraints and Indexes
# ============================================================================
echo "7. Checking constraints and indexes..."

if [ "$BOLT_OK" = true ]; then
    CONSTRAINT_COUNT=$(echo "CALL db.constraints() YIELD description RETURN count(*) as count" | \
        cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" | \
        grep -E "^[0-9]" | tr -d ' ')

    INDEX_COUNT=$(echo "CALL db.indexes() YIELD indexName RETURN count(*) as count" | \
        cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" | \
        grep -E "^[0-9]" | tr -d ' ')

    if [ -n "$CONSTRAINT_COUNT" ]; then
        print_status "OK" "Constraints found: $CONSTRAINT_COUNT"
    else
        print_status "WARNING" "Could not retrieve constraint count"
    fi

    if [ -n "$INDEX_COUNT" ]; then
        print_status "OK" "Indexes found: $INDEX_COUNT"
    else
        print_status "WARNING" "Could not retrieve index count"
    fi

    # Check for required vector indices
    VECTOR_INDEX=$(echo "SHOW INDEXES WHERE name = 'entity_embeddings'" | \
        cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" | \
        grep -c "entity_embeddings" || echo "0")

    if [ "$VECTOR_INDEX" -gt 0 ]; then
        print_status "OK" "Vector index 'entity_embeddings' exists"
    else
        print_status "WARNING" "Vector index 'entity_embeddings' not found"
    fi
else
    print_status "SKIP" "Cannot check indexes (Bolt connection failed)"
fi

echo ""

# ============================================================================
# CHECK 8: APOC Plugin
# ============================================================================
echo "8. Checking APOC plugin availability..."

if [ "$BOLT_OK" = true ]; then
    APOC_CHECK=$(echo "RETURN apoc.version() as version" | \
        cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" 2>&1)

    if echo "$APOC_CHECK" | grep -q "Unknown function"; then
        print_status "FAIL" "APOC plugin is not installed or not enabled"
    elif echo "$APOC_CHECK" | grep -qE "^[0-9]"; then
        print_status "OK" "APOC plugin is installed (version: $APOC_CHECK)"
    else
        print_status "WARNING" "Could not determine APOC status"
    fi
else
    print_status "SKIP" "Cannot check APOC (Bolt connection failed)"
fi

echo ""

# ============================================================================
# CHECK 9: Memory Usage
# ============================================================================
echo "9. Checking memory usage..."

if [ "$BOLT_OK" = true ]; then
    JVM_MEM=$(echo "CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=VM') YIELD attributes RETURN attributes.HeapMemoryUsage.used as heapUsed" | \
        cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" | \
        grep -E "^[0-9]" | tr -d ' ')

    if [ -n "$JVM_MEM" ]; then
        # Convert bytes to MB
        HEAP_MB=$((JVM_MEM / 1024 / 1024))
        print_status "OK" "Heap memory usage: ${HEAP_MB}MB"
    else
        print_status "WARNING" "Could not retrieve memory usage"
    fi
else
    print_status "SKIP" "Cannot check memory (Bolt connection failed)"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "================================================"
echo "Health Check Summary"
echo "================================================"

OVERALL_STATUS="OK"

if [ "$ALL_COMMANDS_OK" = false ] || [ "$ENV_VARS_OK" = false ] || [ "$HTTP_OK" = false ] || [ "$BOLT_OK" = false ]; then
    OVERALL_STATUS="FAIL"
elif [ "$HTTP_CODE" != "200" ] && [ "$HTTP_CODE" != "302" ]; then
    OVERALL_STATUS="WARNING"
fi

if [ "$OVERALL_STATUS" = "OK" ]; then
    echo -e "${GREEN}Overall Status: HEALTHY${NC}"
    echo ""
    echo "Neo4j is running and ready for use!"
    exit 0
elif [ "$OVERALL_STATUS" = "WARNING" ]; then
    echo -e "${YELLOW}Overall Status: WARNING${NC}"
    echo ""
    echo "Neo4j is running but some checks reported warnings."
    echo "Review the output above for details."
    exit 0
else
    echo -e "${RED}Overall Status: UNHEALTHY${NC}"
    echo ""
    echo "Neo4j has critical issues. Please review the errors above."
    exit 1
fi

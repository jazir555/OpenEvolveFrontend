#!/bin/bash

###############################################################################
# Unified Knowledge Query Probe Script
#
# Federation Constitution - Law of Runtime Truth:
# "Before implementing a feature, you must write a probe script that executes
#  the call against the live container. If the probe fails, the feature does
#  not exist."
#
# This script verifies that all underlying systems are accessible
###############################################################################

set -e  # Fail on error
set -u  # Fail on undefined variables

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Unified Knowledge Query - System Probe Suite            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Track results
RAGBITS_OK=false
GRAPHITI_OK=false
VECTORDB_OK=false

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Probing Knowledge Systems..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Probe RAGBits
echo "1️⃣  RAGBits System"
echo "────────────────────────────────────────────────────────────────"
if [ -f "$(dirname "$0")/check_ragbits.sh" ]; then
    if bash "$(dirname "$0")/check_ragbits.sh"; then
        RAGBITS_OK=true
        print_success "RAGBits probe PASSED"
    else
        print_error "RAGBits probe FAILED"
    fi
else
    print_warning "RAGBits probe script not found"
fi
echo ""

# Probe Graphiti
echo "2️⃣  Graphiti System"
echo "────────────────────────────────────────────────────────────────"
if [ -f "$(dirname "$0")/check_graphiti.sh" ]; then
    if bash "$(dirname "$0")/check_graphiti.sh"; then
        GRAPHITI_OK=true
        print_success "Graphiti probe PASSED"
    else
        print_error "Graphiti probe FAILED"
    fi
else
    print_warning "Graphiti probe script not found"
fi
echo ""

# Probe Vector DB
echo "3️⃣  Vector DB System"
echo "────────────────────────────────────────────────────────────────"
if [ -f "$(dirname "$0")/check_vectordb.sh" ]; then
    if bash "$(dirname "$0")/check_vectordb.sh"; then
        VECTORDB_OK=true
        print_success "Vector DB probe PASSED"
    else
        print_error "Vector DB probe FAILED"
    fi
else
    print_warning "Vector DB probe script not found"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Probe Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$RAGBITS_OK" = true ]; then
    print_success "RAGBits:      OPERATIONAL"
else
    print_error "RAGBits:       OFFLINE"
fi

if [ "$GRAPHITI_OK" = true ]; then
    print_success "Graphiti:     OPERATIONAL"
else
    print_error "Graphiti:      OFFLINE"
fi

if [ "$VECTORDB_OK" = true ]; then
    print_success "Vector DB:    OPERATIONAL"
else
    print_error "Vector DB:     OFFLINE"
fi

echo ""

# Calculate overall status
TOTAL=3
OPERATIONAL=0

[ "$RAGBITS_OK" = true ] && ((OPERATIONAL++))
[ "$GRAPHITI_OK" = true ] && ((OPERATIONAL++))
[ "$VECTORDB_OK" = true ] && ((OPERATIONAL++))

echo "Systems Operational: $OPERATIONAL / $TOTAL"
echo ""

if [ "$OPERATIONAL" -eq "$TOTAL" ]; then
    print_success "ALL SYSTEMS OPERATIONAL"
    echo ""
    echo "Unified Knowledge Query Engine is ready for deployment."
    exit 0
elif [ "$OPERATIONAL" -gt 0 ]; then
    print_warning "PARTIAL SYSTEM AVAILABILITY"
    echo ""
    echo "Some systems are offline. The engine will operate in degraded mode."
    exit 0
else
    print_error "ALL SYSTEMS OFFLINE"
    echo ""
    echo "CRITICAL: No knowledge systems are available."
    echo "The engine cannot function without at least one system."
    exit 1
fi

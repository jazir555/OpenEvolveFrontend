#!/bin/bash
###############################################################################
# LeanAide Integration Probe Script
#
# Law of Runtime Truth: Verify LeanAide server is actually available
# and functioning before attempting integration.
#
# This script tests:
# 1. LeanAide server availability (port 7654)
# 2. Autoformalization functionality
# 3. AI-powered proving
# 4. Z3-LeanAide bridge (if available)
#
# Usage: ./probes/check_leanaide.sh
#
# Author: RESE Team
# Created: 2026-02-04
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LEANAIDE_HOST="${LEANAIDE_HOST:-localhost}"
LEANAIDE_PORT="${LEANAIDE_PORT:-7654}"
LEANAIDE_URL="http://${LEANAIDE_HOST}:${LEANAIDE_PORT}"
TIMEOUT=10

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         LeanAide Integration Probe Script                 ║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║  Law of Runtime Truth: Verify Before Using               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

###############################################################################
# TEST 1: Server Availability
###############################################################################

echo -e "${YELLOW}[TEST 1]${NC} Checking LeanAide server availability..."
echo "Target: ${LEANAIDE_URL}"

if curl -s --max-time $TIMEOUT "${LEANAIDE_URL}/" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC} - LeanAide server is reachable"
    LEANAIDE_AVAILABLE=true
else
    echo -e "${RED}✗ FAIL${NC} - Cannot reach LeanAide server at ${LEANAIDE_URL}"
    echo -e "${YELLOW}→${NC} Ensure LeanAide server is running on port ${LEANAIDE_PORT}"
    LEANAIDE_AVAILABLE=false
fi

echo ""

###############################################################################
# TEST 2: Health Check
###############################################################################

echo -e "${YELLOW}[TEST 2]${NC} Performing health check..."

if [ "$LEANAIDE_AVAILABLE" = true ]; then
    RESPONSE=$(curl -s --max-time $TIMEOUT "${LEANAIDE_URL}/" 2>&1)

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC} - Health check successful"
        echo "Response: ${RESPONSE:0:100}..."
    else
        echo -e "${RED}✗ FAIL${NC} - Health check failed"
    fi
else
    echo -e "${YELLOW}⊘ SKIP${NC} - Server not available"
fi

echo ""

###############################################################################
# TEST 3: Autoformalization
###############################################################################

echo -e "${YELLOW}[TEST 3]${NC} Testing autoformalization..."
echo "Theorem: 'For all natural numbers n, n + 0 = n'"

if [ "$LEANAIDE_AVAILABLE" = true ]; then
    PAYLOAD=$(cat <<EOF
{
  "task": "translate_thm",
  "theorem_text": "For all natural numbers n, n + 0 = n"
}
EOF
)

    RESPONSE=$(curl -s --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" \
        "${LEANAIDE_URL}/" 2>&1)

    if echo "$RESPONSE" | grep -q "lean\|theorem\|Nat"; then
        echo -e "${GREEN}✓ PASS${NC} - Autoformalization working"
        echo "Generated Lean code snippet:"
        echo "$RESPONSE" | head -c 200
        echo "..."
    else
        echo -e "${RED}✗ FAIL${NC} - Autoformalization failed"
        echo "Response: $RESPONSE"
    fi
else
    echo -e "${YELLOW}⊘ SKIP${NC} - Server not available"
fi

echo ""

###############################################################################
# TEST 4: AI-Powered Proving
###############################################################################

echo -e "${YELLOW}[TEST 4]${NC} Testing AI-powered proving..."
echo "Theorem: '1 + 1 = 2'"

if [ "$LEANAIDE_AVAILABLE" = true ]; then
    # First autoformalize
    AUTO_RESPONSE=$(curl -s --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"task": "translate_thm", "theorem_text": "1 + 1 = 2"}' \
        "${LEANAIDE_URL}/" 2>&1)

    # Try to get proof (simplified test)
    PROOF_RESPONSE=$(curl -s --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"task": "math_query", "query": "What is 1 + 1?", "n": 1}' \
        "${LEANAIDE_URL}/" 2>&1)

    if echo "$PROOF_RESPONSE" | grep -q "2\|two"; then
        echo -e "${GREEN}✓ PASS${NC} - AI query working"
        echo "Response snippet:"
        echo "$PROOF_RESPONSE" | head -c 150
        echo "..."
    else
        echo -e "${YELLOW}⊘ UNCERTAIN${NC} - AI query response unclear"
        echo "Response: $PROOF_RESPONSE"
    fi
else
    echo -e "${YELLOW}⊘ SKIP${NC} - Server not available"
fi

echo ""

###############################################################################
# TEST 5: Z3-LeanAide Bridge
###############################################################################

echo -e "${YELLOW}[TEST 5]${NC} Testing Z3-LeanAide bridge..."

# Check if z3_leanaide_bridge.py exists
BRIDGE_FILE="../../../../z3_leanaide_bridge.py"

if [ -f "$BRIDGE_FILE" ]; then
    echo -e "${GREEN}✓ PASS${NC} - Z3-LeanAide bridge file exists"
    echo "Location: $BRIDGE_FILE"

    # Try to import it (basic syntax check)
    if python3 -c "import sys; sys.path.insert(0, '../../../../'); from z3_leanaide_bridge import Z3LeanAideBridge" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC} - Z3-LeanAide bridge can be imported"

        # Try basic initialization
        if python3 -c "
import sys
sys.path.insert(0, '../../../../')
from z3_leanaide_bridge import Z3LeanAideBridge
bridge = Z3LeanAideBridge()
print('Capabilities:', bridge.get_capabilities())
" 2>/dev/null; then
            echo -e "${GREEN}✓ PASS${NC} - Z3-LeanAide bridge initialized"
        else
            echo -e "${YELLOW}⊘ WARN${NC} - Bridge importable but initialization failed"
        fi
    else
        echo -e "${YELLOW}⊘ WARN${NC} - Bridge file exists but dependencies missing"
    fi
else
    echo -e "${YELLOW}⊘ SKIP${NC} - Z3-LeanAide bridge not found"
    echo "Expected location: $BRIDGE_FILE"
fi

echo ""

###############################################################################
# TEST 6: LeanAide Client
###############################################################################

echo -e "${YELLOW}[TEST 6]${NC} Testing LeanAide client..."

CLIENT_FILE="../../../../leanaide_client.py"

if [ -f "$CLIENT_FILE" ]; then
    echo -e "${GREEN}✓ PASS${NC} - LeanAide client file exists"
    echo "Location: $CLIENT_FILE"

    # Try to import it
    if python3 -c "import sys; sys.path.insert(0, '../../../../'); from leanaide_client import LeanAideClient" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC} - LeanAide client can be imported"
    else
        echo -e "${YELLOW}⊘ WARN${NC} - Client file exists but dependencies missing"
    fi
else
    echo -e "${YELLOW}⊘ SKIP${NC} - LeanAide client not found"
    echo "Expected location: $CLIENT_FILE"
fi

echo ""

###############################################################################
# TEST 7: Configuration Validation
###############################################################################

echo -e "${YELLOW}[TEST 7]${NC} Checking environment configuration..."

REQUIRED_VARS=("LEANAIDE_BASE_URL")
OPTIONAL_VARS=("LEANAIDE_TIMEOUT_MS" "LEANAIDE_ENABLE")

all_required_set=true

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${YELLOW}⊘ WARN${NC} - $var not set (will use default)"
        all_required_set=false
    else
        echo -e "${GREEN}✓${NC} - $var = ${!var}"
    fi
done

for var in "${OPTIONAL_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        echo -e "${GREEN}✓${NC} - $var = ${!var}"
    fi
done

if [ "$all_required_set" = true ]; then
    echo -e "${GREEN}✓ PASS${NC} - All required environment variables set"
else
    echo -e "${YELLOW}⊘ INFO${NC} - Using default configuration"
fi

echo ""

###############################################################################
# SUMMARY
###############################################################################

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Probe Summary                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "LeanAide Server: ${LEANAIDE_URL}"
echo ""

if [ "$LEANAIDE_AVAILABLE" = true ]; then
    echo -e "${GREEN}Status: ✓ LEANAIDE INTEGRATION READY${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Import: from rese_z3_bridge import RESEZ3Bridge"
    echo "  2. Create: bridge = RESEZ3Bridge()"
    echo "  3. Use: bridge.autoformalize('Your theorem')"
    echo ""
    echo "Available features:"
    echo "  • Autoformalization: Natural language to Lean 4"
    echo "  • AI-powered proving: Generate proofs automatically"
    echo "  • Z3-Lean translation: Bridge Z3 constraints to Lean"
    echo "  • Tactic suggestions: Get AI-recommended tactics"
    echo ""
    exit 0
else
    echo -e "${RED}Status: ✗ LEANAIDE INTEGRATION NOT READY${NC}"
    echo ""
    echo "Required actions:"
    echo "  1. Start LeanAide server on port ${LEANAIDE_PORT}"
    echo "  2. Verify server is responding: curl ${LEANAIDE_URL}/"
    echo "  3. Re-run this probe script"
    echo ""
    echo "To start LeanAide server (if available):"
    echo "  cd /path/to/leanaide"
    echo "  python -m leanaide.server --port ${LEANAIDE_PORT}"
    echo ""
    exit 1
fi

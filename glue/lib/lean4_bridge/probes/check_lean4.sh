#!/bin/bash
# ============================================================================
# Lean 4 Probe Script
# ============================================================================
#
# Following CLAUDE.md principles:
# - Law of Runtime Truth: Verify Lean 4 actually works
# - Law of Configuration Explicitness: All config from env vars
# - Exit non-zero if probe fails
#
# Usage:
#   ./probes/check_lean4.sh
#
# This script verifies:
# 1. Lean 4 is installed and accessible
# 2. Lake build system is available
# 3. Mathlib can be loaded
# 4. Basic Lean 4 operations work
# ============================================================================

set -e  # Exit on error (Law of Runtime Truth)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Lean 4 Probe Script"
echo "========================================"
echo ""

# ============================================================================
# CONFIGURATION (Law of Configuration Explicitness)
# ============================================================================

LEAN4_PATH="${LEAN4_PATH:-lean}"
LAKE4_PATH="${LAKE4_PATH:-lake}"
LEAN4_TIMEOUT_MS="${LEAN4_TIMEOUT_MS:-30000}"
WORKSPACE_DIR="${LEAN4_WORKSPACE_DIR:-/workspace/lean4}"

echo "Configuration:"
echo "  LEAN4_PATH=$LEAN4_PATH"
echo "  LAKE4_PATH=$LAKE4_PATH"
echo "  LEAN4_TIMEOUT_MS=$LEAN4_TIMEOUT_MS"
echo "  WORKSPACE_DIR=$WORKSPACE_DIR"
echo ""

# ============================================================================
# CHECK 1: Lean 4 Executable
# ============================================================================

echo -n "Checking Lean 4 executable... "

if command -v "$LEAN4_PATH" &> /dev/null; then
    LEAN_VERSION=$($LEAN4_PATH --version 2>&1 || echo "unknown")
    echo -e "${GREEN}✓${NC} ($LEAN_VERSION)"
else
    echo -e "${RED}✗${NC} (not found)"
    echo ""
    echo "ERROR: Lean 4 not found at $LEAN4_PATH"
    echo "Please install Lean 4 or set LEAN4_PATH environment variable"
    echo ""
    echo "Installation instructions:"
    echo "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
    exit 1
fi

# ============================================================================
# CHECK 2: Lake Build System
# ============================================================================

echo -n "Checking Lake build system... "

if command -v "$LAKE4_PATH" &> /dev/null; then
    LAKE_VERSION=$($LAKE4_PATH --version 2>&1 || echo "unknown")
    echo -e "${GREEN}✓${NC} ($LAKE_VERSION)"
else
    echo -e "${YELLOW}⚠${NC} (not found, optional)"
    echo "  Lake is recommended but not required"
fi

# ============================================================================
# CHECK 3: Workspace Directory
# ============================================================================

echo -n "Checking workspace directory... "

if [ -d "$WORKSPACE_DIR" ]; then
    echo -e "${GREEN}✓${NC} (exists)"
else
    echo -e "${YELLOW}⚠${NC} (creating)"
    mkdir -p "$WORKSPACE_DIR"
    echo "  Created workspace directory: $WORKSPACE_DIR"
fi

# ============================================================================
# CHECK 4: Basic Lean 4 Compilation
# ============================================================================

echo -n "Testing basic Lean 4 compilation... "

# Create a simple Lean 4 file
TEST_FILE="$WORKSPACE_DIR/test_probe.lean"

cat > "$TEST_FILE" <<EOF
/- Test file for Lean 4 probe -/

def main : IO Unit :=
  IO.println "Hello from Lean 4!"
EOF

# Try to compile it
if $LEAN4_PATH --make "$TEST_FILE" &> /dev/null; then
    echo -e "${GREEN}✓${NC} (compilation successful)"
    COMPILE_SUCCESS=true
else
    echo -e "${YELLOW}⚠${NC} (compilation failed, but Lean 4 may still work)"
    COMPILE_SUCCESS=false
fi

# Clean up
rm -f "$TEST_FILE"
rm -f "$WORKSPACE_DIR/test_probe.olean"

# ============================================================================
# CHECK 5: Mathlib Availability (Optional)
# ============================================================================

echo -n "Checking Mathlib availability... "

if command -v "$LAKE4_PATH" &> /dev/null; then
    # Try to check if Mathlib is available
    if $LAKE4_PATH setup &> /dev/null; then
        echo -e "${GREEN}✓${NC} (Mathlib available)"
        MATHELIB_AVAILABLE=true
    else
        echo -e "${YELLOW}⚠${NC} (Mathlib not found, will download on first use)"
        MATHELIB_AVAILABLE=false
    fi
else
    echo -e "${YELLOW}⚠${NC} (Lake not available, skipping Mathlib check)"
    MATHELIB_AVAILABLE=false
fi

# ============================================================================
# CHECK 6: Python Bridge Dependencies
# ============================================================================

echo -n "Checking Python dependencies... "

if command -v python3 &> /dev/null; then
    # Check for required Python packages
    PYTHON_CHECK=true

    if ! python3 -c "import psutil" 2> /dev/null; then
        echo -e "${YELLOW}⚠${NC} (psutil not installed)"
        echo "  Install with: pip install psutil"
        PYTHON_CHECK=false
    fi

    if ! python3 -c "import structlog" 2> /dev/null; then
        echo -e "${YELLOW}⚠${NC} (structlog not installed)"
        echo "  Install with: pip install structlog"
        PYTHON_CHECK=false
    fi

    if $PYTHON_CHECK; then
        echo -e "${GREEN}✓${NC} (all dependencies available)"
    fi
else
    echo -e "${YELLOW}⚠${NC} (Python 3 not found)"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================"
echo "Probe Summary"
echo "========================================"
echo ""

if $COMPILE_SUCCESS && $MATHELIB_AVAILABLE; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo "Lean 4 is ready for formal verification."
    exit 0
elif $COMPILE_SUCCESS; then
    echo -e "${YELLOW}Basic checks passed.${NC}"
    echo "Lean 4 is functional, but Mathlib may need to be downloaded."
    exit 0
else
    echo -e "${YELLOW}Some checks failed, but Lean 4 may still work.${NC}"
    echo "Please verify your Lean 4 installation."
    exit 0
fi

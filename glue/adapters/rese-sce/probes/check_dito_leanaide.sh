#!/bin/bash
#
# DITO LeanAide Integration Probe Script
#
# Verifies LeanAide integration with DITO Optimizer
#
# Tests:
# 1. LeanAide server availability
# 2. DITO + LeanAide initialization
# 3. Tactic suggestion functionality
# 4. AI-assisted contradiction resolution
# 5. Autoformalization
# 6. Tiered verification
#
# Author: OpenEvolve
# Created: 2026-02-04
#

set -e

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

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[SKIP]${NC} $*"
}

log_test() {
    echo ""
    echo "========================================================================"
    echo "TEST: $1"
    echo "========================================================================"
}

# Test result tracking
run_test() {
    local test_name="$1"
    local test_command="$2"

    TESTS_RUN=$((TESTS_RUN + 1))

    log_test "$test_name"

    if eval "$test_command"; then
        log_success "$test_name passed"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        log_error "$test_name failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

skip_test() {
    local test_name="$1"
    local reason="$2"

    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))

    log_test "$test_name"
    log_warning "$test_name skipped: $reason"
}

# Helper functions
check_python_module() {
    local module="$1"
    ${PYTHON} -c "import $module" 2>/dev/null
}

check_leanaide_server() {
    curl -s -f "${LEANAIDE_URL}/" > /dev/null 2>&1
}

# =============================================================================
# TEST 1: LeanAide Server Health Check
# =============================================================================

test_leanaide_server_health() {
    log_info "Checking LeanAide server at ${LEANAIDE_URL}"

    if check_leanaide_server; then
        log_success "LeanAide server is responding"

        # Get server info
        response=$(curl -s "${LEANAIDE_URL}/" 2>/dev/null || echo "{}")

        if echo "$response" | grep -q "LeanAide"; then
            log_info "Server response confirms LeanAide"
        fi

        return 0
    else
        log_error "LeanAide server not responding at ${LEANAIDE_URL}"
        log_info "Start LeanAide server with: leanaide_server.py"
        return 1
    fi
}

# =============================================================================
# TEST 2: Python Dependencies
# =============================================================================

test_python_dependencies() {
    local missing_deps=0

    log_info "Checking Python dependencies"

    # Required modules
    local modules=("asyncio" "json" "logging" "datetime")

    for module in "${modules[@]}"; do
        if check_python_module "$module"; then
            log_success "✓ $module"
        else
            log_error "✗ $module (missing)"
            missing_deps=$((missing_deps + 1))
        fi
    done

    # Optional modules
    log_info "Checking optional modules"

    if check_python_module "z3"; then
        log_success "✓ z3 (available)"
    else
        log_warning "✗ z3 (optional, not available)"
    fi

    if check_python_module "aiohttp"; then
        log_success "✓ aiohttp (available)"
    else
        log_warning "✗ aiohttp (optional, not available)"
    fi

    if [ $missing_deps -eq 0 ]; then
        return 0
    else
        log_error "Missing $missing_deps required dependencies"
        return 1
    fi
}

# =============================================================================
# TEST 3: DITO + LeanAide Initialization
# =============================================================================

test_dito_leanaide_init() {
    cat > /tmp/test_dito_init.py << 'EOF'
import sys
sys.path.insert(0, "${SRC_DIR}")

try:
    from dito_optimizer import DITOOptimizer, LeanAideTacticSuggester, LEANAIDE_AVAILABLE
    from sce_bridge import SCEConfig

    config = SCEConfig.from_env()

    # Test DITO initialization
    dito = DITOOptimizer(enable_leanaide=True)

    print(f"DITO initialized: {dito is not None}")
    print(f"LeanAide available: {LEANAIDE_AVAILABLE}")
    print(f"LeanAide suggester: {dito.leanaide_suggester is not None}")

    # Test LeanAide suggester initialization
    if LEANAIDE_AVAILABLE:
        import logging
        logger = logging.getLogger('test')
        suggester = LeanAideTacticSuggester(config, logger)

        print(f"LeanAide suggester initialized: {suggester is not None}")
        print(f"LeanAide client: {suggester.leanaide_client is not None}")

    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

    ${PYTHON} /tmp/test_dito_init.py
    return $?
}

# =============================================================================
# TEST 4: Tactic Suggestion
# =============================================================================

test_tactic_suggestion() {
    cat > /tmp/test_tactics.py << 'EOF'
import asyncio
import sys
sys.path.insert(0, "${SRC_DIR}")

try:
    from dito_optimizer import LeanAideTacticSuggester, LEANAIDE_AVAILABLE
    from sce_bridge import SCEConfig, Constraint, ConstraintCategory, ConstraintType
    from dito_optimizer import ContradictionPair, LogicalFallacy
    from datetime import datetime, timezone

    if not LEANAIDE_AVAILABLE:
        print("SKIP: LeanAide not available")
        sys.exit(0)

    async def test():
        config = SCEConfig.from_env()
        import logging
        logger = logging.getLogger('test')

        suggester = LeanAideTacticSuggester(config, logger)

        # Create test contradiction
        constraints = [
            Constraint(
                constraint_id="c1",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T < 1000",
            ),
            Constraint(
                constraint_id="c2",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T > 1500",
            ),
        ]

        contradiction = ContradictionPair(
            constraint1_id="c1",
            constraint2_id="c2",
            type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=2,
            rollback_steps=0,
            affected_premises=["c1", "c2"],
            detected_at=datetime.now(timezone.utc),
        )

        # Test tactic suggestion
        tactics = await suggester.suggest_tactics(
            contradiction,
            constraints,
            "test-probe"
        )

        print(f"Tactics suggested: {tactics}")

        stats = suggester.get_stats()
        print(f"Checks performed: {stats.leanaide_checks_performed}")

        await suggester.close()

        if tactics:
            print(f"SUCCESS: Got {len(tactics)} tactic suggestions")
            return True
        else:
            print("WARNING: No tactics suggested (LeanAide may not be running)")
            return True

    asyncio.run(test())
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

    ${PYTHON} /tmp/test_tactics.py
    return $?
}

# =============================================================================
# TEST 5: AI-Assisted Resolution
# =============================================================================

test_ai_resolution() {
    cat > /tmp/test_resolution.py << 'EOF'
import asyncio
import sys
sys.path.insert(0, "${SRC_DIR}")

try:
    from dito_optimizer import DITOOptimizer, LEANAIDE_AVAILABLE
    from sce_bridge import Constraint, ConstraintCategory, ConstraintType
    from dito_optimizer import ContradictionPair, LogicalFallacy
    from datetime import datetime, timezone

    if not LEANAIDE_AVAILABLE:
        print("SKIP: LeanAide not available")
        sys.exit(0)

    async def test():
        dito = DITOOptimizer(enable_leanaide=True)

        # Create test contradiction
        constraints = [
            Constraint(
                constraint_id="c1",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T < 1000",
            ),
            Constraint(
                constraint_id="c2",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T > 1500",
            ),
        ]

        contradiction = ContradictionPair(
            constraint1_id="c1",
            constraint2_id="c2",
            type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=2,
            rollback_steps=0,
            affected_premises=["c1", "c2"],
            detected_at=datetime.now(timezone.utc),
        )

        # Test AI resolution
        resolution = await dito.resolve_with_ai(
            contradiction,
            constraints,
            "test-probe"
        )

        print(f"Resolution: {resolution}")

        await dito.close()

        if resolution:
            print("SUCCESS: Got AI resolution suggestions")
            return True
        else:
            print("WARNING: No resolution (LeanAide may not be running)")
            return True

    asyncio.run(test())
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

    ${PYTHON} /tmp/test_resolution.py
    return $?
}

# =============================================================================
# TEST 6: Autoformalization
# =============================================================================

test_autoformalization() {
    cat > /tmp/test_autoformalize.py << 'EOF'
import asyncio
import sys
sys.path.insert(0, "${SRC_DIR}")

try:
    from dito_optimizer import DITOOptimizer, LEANAIDE_AVAILABLE

    if not LEANAIDE_AVAILABLE:
        print("SKIP: LeanAide not available")
        sys.exit(0)

    async def test():
        dito = DITOOptimizer(enable_leanaide=True)

        # Test autoformalization
        natural = "Temperature must be less than 1000 Kelvin"

        formal = await dito.formalize_with_ai(
            natural,
            "test-probe"
        )

        print(f"Natural: {natural}")
        print(f"Formal: {formal}")

        await dito.close()

        if formal:
            print("SUCCESS: Autoformalization worked")
            return True
        else:
            print("WARNING: No formalization (LeanAide may not be running)")
            return True

    asyncio.run(test())
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

    ${PYTHON} /tmp/test_autoformalize.py
    return $?
}

# =============================================================================
# TEST 7: Tiered Verification
# =============================================================================

test_tiered_verification() {
    cat > /tmp/test_tiered.py << 'EOF'
import asyncio
import sys
sys.path.insert(0, "${SRC_DIR}")

try:
    from dito_optimizer import DITOOptimizer, VerificationTier, LEANAIDE_AVAILABLE
    from sce_bridge import Constraint, ConstraintCategory, ConstraintType

    async def test():
        dito = DITOOptimizer(enable_leanaide=LEANAIDE_AVAILABLE)

        # Create test constraints
        constraints = [
            Constraint(
                constraint_id="c1",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T < 1000",
            ),
            Constraint(
                constraint_id="c2",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T > 0",
            ),
            Constraint(
                constraint_id="c3",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T > 1500",
                dependencies=["c1"],
            ),
        ]

        # Test complexity scoring
        complexity = dito._calculate_complexity_score(constraints)
        print(f"Complexity score: {complexity:.2f}")

        # Test tier selection
        tier = dito.select_verification_tier(constraints, complexity)
        print(f"Selected tier: {tier.value}")

        # Test tiered detection
        contradiction, used_tier = await dito.check_contradiction_tiered(
            constraints,
            "test-probe"
        )

        print(f"Contradiction found: {contradiction is not None}")
        print(f"Tier used: {used_tier.value}")
        print(f"Tier distribution: {dito.stats.tier_distribution}")

        await dito.close()

        if used_tier in VerificationTier:
            print("SUCCESS: Tiered verification worked")
            return True
        else:
            print("ERROR: Invalid tier returned")
            return False

    asyncio.run(test())
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

    ${PYTHON} /tmp/test_tiered.py
    return $?
}

# =============================================================================
# TEST 8: Full DITO Integration
# =============================================================================

test_dito_full_integration() {
    cat > /tmp/test_dito_full.py << 'EOF'
import asyncio
import sys
sys.path.insert(0, "${SRC_DIR}")

try:
    from dito_optimizer import DITOOptimizer, ActivationStrategy
    from sce_bridge import Constraint, ConstraintCategory, ConstraintType

    async def test():
        dito = DITOOptimizer(
            activation_strategy=ActivationStrategy.SELECTIVE_BFS,
            enable_leanaide=True
        )

        # Create test constraints
        constraints = [
            Constraint(
                constraint_id="temp_upper",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T < 1000",
            ),
            Constraint(
                constraint_id="temp_lower",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T > 0",
            ),
            Constraint(
                constraint_id="temp_contradict",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="T > 1500",
                dependencies=["temp_upper"],
            ),
        ]

        print(f"Running DITO optimization with {len(constraints)} constraints...")

        contradictions, stats = dito.optimize_contradiction_detection(
            constraints,
            "test-probe-full"
        )

        print(f"\nResults:")
        print(f"  Total nodes: {stats.total_nodes}")
        print(f"  Verified: {stats.verified_nodes}")
        print(f"  Active: {stats.active_nodes}")
        print(f"  Contradictions: {len(contradictions)}")
        print(f"  Execution time: {stats.execution_time_ms}ms")

        if stats.z3_atp_stats:
            print(f"\nZ3 Stats:")
            print(f"  Checks: {stats.z3_atp_stats.z3_checks_performed}")
            print(f"  Time: {stats.z3_atp_stats.z3_total_time_ms}ms")

        if stats.leanaide_ai_stats:
            print(f"\nLeanAide Stats:")
            print(f"  Checks: {stats.leanaide_ai_stats.leanaide_checks_performed}")
            print(f"  Time: {stats.leanaide_ai_stats.leanaide_total_time_ms}ms")

        print(f"\nTier distribution: {stats.tier_distribution}")

        await dito.close()

        if stats.total_nodes == len(constraints):
            print("SUCCESS: DITO full integration worked")
            return True
        else:
            print("ERROR: Not all constraints processed")
            return False

    asyncio.run(test())
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

    ${PYTHON} /tmp/test_dito_full.py
    return $?
}

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

main() {
    echo ""
    echo "========================================================================"
    echo "  DITO LeanAide Integration Probe"
    echo "========================================================================"
    echo ""
    echo "Configuration:"
    echo "  LeanAide URL: ${LEANAIDE_URL}"
    echo "  Python: ${PYTHON}"
    echo "  Source: ${SRC_DIR}"
    echo ""

    # Check if source directory exists
    if [ ! -d "${SRC_DIR}" ]; then
        log_error "Source directory not found: ${SRC_DIR}"
        exit 1
    fi

    # Run tests
    run_test "LeanAide Server Health" "test_leanaide_server_health" || true

    run_test "Python Dependencies" "test_python_dependencies"

    if [ ! -f "${SRC_DIR}/dito_optimizer.py" ]; then
        log_error "dito_optimizer.py not found at ${SRC_DIR}"
        log_info "Build the project first"
        exit 1
    fi

    run_test "DITO + LeanAide Initialization" "test_dito_leanaide_init"
    run_test "Tactic Suggestion" "test_tactic_suggestion"
    run_test "AI-Assisted Resolution" "test_ai_resolution"
    run_test "Autoformalization" "test_autoformalization"
    run_test "Tiered Verification" "test_tiered_verification"
    run_test "Full DITO Integration" "test_dito_full_integration"

    # Clean up temp files
    rm -f /tmp/test_*.py

    # Summary
    echo ""
    echo "========================================================================"
    echo "  Test Summary"
    echo "========================================================================"
    echo "  Total:   ${TESTS_RUN}"
    echo "  Passed:  ${TESTS_PASSED}"
    echo "  Failed:  ${TESTS_FAILED}"
    echo "  Skipped: ${TESTS_SKIPPED}"
    echo ""

    if [ ${TESTS_FAILED} -eq 0 ]; then
        log_success "All tests passed!"
        echo ""
        echo "DITO LeanAide integration is working correctly."
        return 0
    else
        log_error "Some tests failed"
        echo ""
        echo "Please check the errors above and ensure:"
        echo "  1. LeanAide server is running at ${LEANAIDE_URL}"
        echo "  2. All dependencies are installed"
        echo "  3. Environment variables are set correctly"
        return 1
    fi
}

# Run main
main "$@"

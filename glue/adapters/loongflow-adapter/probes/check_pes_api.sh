#!/bin/bash

###############################################################################
# check_pes_api.sh - Probe LoongFlow PES Agent Framework
#
# This script verifies that the PES (Plan-Execute-Summary) agent framework
# is accessible and can be instantiated.
#
# Environment Variables Required:
#   PYTHONPATH - Must include LoongFlow src directory
#
# Usage:
#   ./check_pes_api.sh
#
# Exit Codes:
#   0 - PES framework is accessible
#   1 - PES framework is not accessible
###############################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-}$(date -u +"%Y-%m-%dT%H:%M:%SZ") [check_pes_api] $1${NC}"
}

log_info() {
    log "$1" ""
}

log_success() {
    log "$1" "$GREEN"
}

log_error() {
    log "$1" "$RED"
}

log_warning() {
    log "$1" "$YELLOW"
}

# Check Python environment
check_python_env() {
    log_info "Checking Python environment..."

    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        return 1
    fi

    local python_version
    python_version=$(python3 --version)
    log_success "Python found: $python_version"

    # Set PYTHONPATH to include LoongFlow
    local loongflow_path="../../../../core-projects/LoongFlow"
    export PYTHONPATH="${PYTHONPATH:-}:${loongflow_path}/src"

    log_info "PYTHONPATH: $PYTHONPATH"

    return 0
}

# Test PES Agent import
check_pes_agent_import() {
    log_info "Testing PES Agent import..."

    local test_script="
import sys
sys.path.insert(0, '../../../../core-projects/LoongFlow/src')

try:
    from loongflow.framework.pes.pes_agent import PESAgent
    print('SUCCESS: PESAgent imported successfully')
    print('PESAgent class:', PESAgent)
    sys.exit(0)
except Exception as e:
    print('ERROR: Failed to import PESAgent:', str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    if python3 -c "$test_script"; then
        log_success "PES Agent import test passed"
        return 0
    else
        log_error "PES Agent import test failed"
        return 1
    fi
}

# Test EvolveDatabase import
check_database_import() {
    log_info "Testing EvolveDatabase import..."

    local test_script="
import sys
sys.path.insert(0, '../../../../core-projects/LoongFlow/src')

try:
    from loongflow.framework.pes.database.database import EvolveDatabase
    print('SUCCESS: EvolveDatabase imported successfully')
    print('EvolveDatabase class:', EvolveDatabase)
    sys.exit(0)
except Exception as e:
    print('ERROR: Failed to import EvolveDatabase:', str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    if python3 -c "$test_script"; then
        log_success "EvolveDatabase import test passed"
        return 0
    else
        log_error "EvolveDatabase import test failed"
        return 1
    fi
}

# Test Context import
check_context_import() {
    log_info "Testing Context import..."

    local test_script="
import sys
sys.path.insert(0, '../../../../core-projects/LoongFlow/src')

try:
    from loongflow.framework.pes.context import Context
    print('SUCCESS: Context imported successfully')
    print('Context class:', Context)
    sys.exit(0)
except Exception as e:
    print('ERROR: Failed to import Context:', str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    if python3 -c "$test_script"; then
        log_success "Context import test passed"
        return 0
    else
        log_error "Context import test failed"
        return 1
    fi
}

# Test Solution model import
check_solution_import() {
    log_info "Testing Solution model import..."

    local test_script="
import sys
sys.path.insert(0, '../../../../core-projects/LoongFlow/src')

try:
    from loongflow.agentsdk.memory.evolution.base_memory import Solution
    print('SUCCESS: Solution imported successfully')
    print('Solution class:', Solution)
    
    # Try to instantiate with minimal fields
    sol = Solution(
        solution='test_solution',
        evaluation='test_evaluation',
        score=0.5,
        island_id=0,
        generate_plan='test_plan',
        summary='test_summary'
    )
    print('Solution instance created:', sol)
    sys.exit(0)
except Exception as e:
    print('ERROR: Failed to import/create Solution:', str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    if python3 -c "$test_script"; then
        log_success "Solution model test passed"
        return 0
    else
        log_error "Solution model test failed"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting LoongFlow PES API probe..."
    echo ""

    local exit_code=0

    if ! check_python_env; then
        exit_code=1
    fi
    echo ""

    if ! check_pes_agent_import; then
        exit_code=1
    fi
    echo ""

    if ! check_database_import; then
        exit_code=1
    fi
    echo ""

    if ! check_context_import; then
        exit_code=1
    fi
    echo ""

    if ! check_solution_import; then
        exit_code=1
    fi
    echo ""

    if [[ $exit_code -eq 0 ]]; then
        log_success "LoongFlow PES API probe completed successfully"
    else
        log_error "LoongFlow PES API probe failed"
    fi

    exit $exit_code
}

# Run main function
main "$@"

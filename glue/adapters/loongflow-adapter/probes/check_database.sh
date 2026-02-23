#!/bin/bash

###############################################################################
# check_database.sh - Probe LoongFlow EvolveDatabase Operations
#
# This script tests the EvolveDatabase API operations including:
# - sample_solution()
# - add_solution()
# - update_solution()
# - get_best_solutions()
# - save_checkpoint()
#
# Usage:
#   ./check_database.sh
#
# Exit Codes:
#   0 - Database operations work correctly
#   1 - Database operations failed
###############################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-}$(date -u +"%Y-%m-%dT%H:%M:%SZ") [check_database] $1${NC}"
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

# Test database operations
check_database_operations() {
    log_info "Testing EvolveDatabase operations..."

    local test_script="
import sys
import asyncio
sys.path.insert(0, '../../../../core-projects/LoongFlow/src')

async def test_database():
    try:
        from loongflow.framework.pes.database.database import EvolveDatabase
        from loongflow.framework.pes.context.config import DatabaseConfig
        from loongflow.agentsdk.memory.evolution.base_memory import Solution
        
        print('Creating in-memory database...')
        config = DatabaseConfig(
            storage_type='in_memory',
            num_islands=1,
            population_size=10,
            checkpoint_interval=1,
        )
        db = EvolveDatabase(config)
        print('Database created:', db)
        
        print('\nTesting sample_solution()...')
        sample = db.sample_solution()
        print('Sample result:', sample)
        
        print('\nTesting add_solution()...')
        solution = Solution(
            solution='test_solution_code',
            evaluation='test_evaluation',
            score=0.75,
            island_id=0,
            generate_plan='test_plan',
            summary='test_summary'
        )
        solution_id = await db.add_solution(solution)
        print('Solution added with ID:', solution_id)
        
        print('\nTesting update_solution()...')
        await db.update_solution(solution_id, score=0.85)
        print('Solution updated')
        
        print('\nTesting get_best_solutions()...')
        best = db.get_best_solutions(top_k=5)
        print('Best solutions:', len(best), 'solutions')
        
        print('\nTesting memory_status()...')
        status = db.memory_status()
        print('Memory status:', status)
        
        print('\nAll database operations completed successfully!')
        return 0
        
    except Exception as e:
        print('ERROR: Database operations failed:', str(e))
        import traceback
        traceback.print_exc()
        return 1

exit_code = asyncio.run(test_database())
sys.exit(exit_code)
"

    if python3 -c "$test_script"; then
        log_success "Database operations test passed"
        return 0
    else
        log_error "Database operations test failed"
        return 1
    fi
}

# Test checkpoint save/load
check_checkpoint_operations() {
    log_info "Testing checkpoint save/load operations..."

    local test_script="
import sys
import asyncio
import tempfile
import shutil
sys.path.insert(0, '../../../../core-projects/LoongFlow/src')

async def test_checkpoints():
    try:
        from loongflow.framework.pes.database.database import EvolveDatabase
        from loongflow.framework.pes.context.config import DatabaseConfig
        from loongflow.agentsdk.memory.evolution.base_memory import Solution
        
        # Create temp directory for checkpoints
        temp_dir = tempfile.mkdtemp(prefix='loongflow_test_')
        print('Using temp directory:', temp_dir)
        
        try:
            print('Creating database...')
            config = DatabaseConfig(
                storage_type='in_memory',
                num_islands=1,
                population_size=10,
                checkpoint_interval=1,
                output_path=temp_dir,
            )
            db = EvolveDatabase(config)
            
            print('Adding test solution...')
            solution = Solution(
                solution='test_solution_for_checkpoint',
                evaluation='test_eval',
                score=0.9,
                island_id=0,
                generate_plan='test_plan',
                summary='test_summary'
            )
            await db.add_solution(solution)
            
            print('\nTesting save_checkpoint()...')
            await db.save_checkpoint(temp_dir, 'test-checkpoint')
            print('Checkpoint saved')
            
            print('\nTesting load_checkpoint()...')
            db.load_checkpoint(temp_dir + '/test-checkpoint')
            print('Checkpoint loaded')
            
            print('\nCheckpoint operations completed successfully!')
            return 0
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            print('\nCleaned up temp directory')
            
    except Exception as e:
        print('ERROR: Checkpoint operations failed:', str(e))
        import traceback
        traceback.print_exc()
        return 1

exit_code = asyncio.run(test_checkpoints())
sys.exit(exit_code)
"

    if python3 -c "$test_script"; then
        log_success "Checkpoint operations test passed"
        return 0
    else
        log_error "Checkpoint operations test failed"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting LoongFlow Database probe..."
    echo ""

    local exit_code=0

    if ! check_database_operations; then
        exit_code=1
    fi
    echo ""

    if ! check_checkpoint_operations; then
        exit_code=1
    fi
    echo ""

    if [[ $exit_code -eq 0 ]]; then
        log_success "LoongFlow Database probe completed successfully"
    else
        log_error "LoongFlow Database probe failed"
    fi

    exit $exit_code
}

# Run main function
main "$@"

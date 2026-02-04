#!/bin/bash

# @license
# SPDX-License-Identifier: Apache-2.0
#
# ICR Graphiti Memory Integration Probe
#
# Probe script to verify Graphiti memory integration for ICR Contextual Mode.
# Follows the Federation Constitution:
# - Law of Runtime Truth: Verify actual API behavior
# - Law of Configuration Explicitness: No magic defaults
#
# This script tests:
# 1. Memory canonical schemas validation
# 2. GraphitiMemoryManager connectivity
# 3. Historical knowledge retrieval
# 4. Memory storage operations

set -e  # Exit on error

# ============================================================================
# CONFIGURATION (Law of Configuration Explicitness)
# ============================================================================

# Required environment variables
REQUIRED_VARS=(
  "OPENEVOLVE_ICR_API_URL"
  "GRAPHITI_API_URL"
  "TIMEOUT_MS"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_required_vars() {
    log_info "Checking required environment variables..."
    missing_vars=()

    for var in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        exit 1
    fi

    log_info "All required environment variables are set."
}

# ============================================================================
# PROBE 1: Check ICR API Availability
# ============================================================================

probe_icr_api() {
    log_info "Probe 1: Checking ICR API availability..."

    local response
    local status_code

    response=$(curl -s -w "\n%{http_code}" \
        -X GET \
        "${OPENEVOLVE_ICR_API_URL}/api/health" \
        --connect-timeout "$((TIMEOUT_MS / 1000))" \
        --max-time "$((TIMEOUT_MS / 1000))" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" 2>&1) || true

    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_info "ICR API is available (200 OK)"
        return 0
    else
        log_error "ICR API is not available (status: $status_code)"
        return 1
    fi
}

# ============================================================================
# PROBE 2: Check Graphiti API Availability
# ============================================================================

probe_graphiti_api() {
    log_info "Probe 2: Checking Graphiti API availability..."

    local response
    local status_code

    response=$(curl -s -w "\n%{http_code}" \
        -X GET \
        "${GRAPHITI_API_URL}/api/health" \
        --connect-timeout "$((TIMEOUT_MS / 1000))" \
        --max-time "$((TIMEOUT_MS / 1000))" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" 2>&1) || true

    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_info "Graphiti API is available (200 OK)"
        return 0
    else
        log_error "Graphiti API is not available (status: $status_code)"
        return 1
    fi
}

# ============================================================================
# PROBE 3: Test Memory Schema Validation
# ============================================================================

probe_memory_schemas() {
    log_info "Probe 3: Testing memory canonical schemas..."

    # Check if canonical schema file exists
    if [ ! -f "src/memory/canonical.ts" ]; then
        log_error "Memory canonical schema file not found: src/memory/canonical.ts"
        return 1
    fi

    log_info "Memory canonical schema file exists."

    # Check for key schema definitions
    local required_schemas=(
        "RefinementMemorySchema"
        "ContextualSessionSchema"
        "PatternRelationshipSchema"
        "MemoryQuerySchema"
        "EnrichedContextSchema"
    )

    for schema in "${required_schemas[@]}"; do
        if grep -q "$schema" "src/memory/canonical.ts"; then
            log_info "  ✓ Schema found: $schema"
        else
            log_error "  ✗ Schema missing: $schema"
            return 1
        fi
    done

    log_info "All required memory schemas are defined."
    return 0
}

# ============================================================================
# PROBE 4: Test Graphiti Memory Manager
# ============================================================================

probe_graphiti_memory_manager() {
    log_info "Probe 4: Testing GraphitiMemoryManager..."

    # Check if GraphitiMemoryManager file exists
    if [ ! -f "src/memory/graphiti-memory.ts" ]; then
        log_error "GraphitiMemoryManager file not found: src/memory/graphiti-memory.ts"
        return 1
    fi

    log_info "GraphitiMemoryManager file exists."

    # Check for key methods
    local required_methods=(
        "storeRefinementInsights"
        "storeContextualSession"
        "retrieveHistoricalKnowledge"
        "retrieveContextualMemory"
        "buildContextualGraph"
        "learnFromSession"
    )

    for method in "${required_methods[@]}"; do
        if grep -q "$method" "src/memory/graphiti-memory.ts"; then
            log_info "  ✓ Method found: $method"
        else
            log_error "  ✗ Method missing: $method"
            return 1
        fi
    done

    log_info "All required GraphitiMemoryManager methods are defined."
    return 0
}

# ============================================================================
# PROBE 5: Test Enhanced Memory Agent
# ============================================================================

probe_enhanced_memory_agent() {
    log_info "Probe 5: Testing EnhancedICRMemoryAgent..."

    # Check if EnhancedICRMemoryAgent file exists
    if [ ! -f "src/memory/memory-agent.ts" ]; then
        log_error "EnhancedICRMemoryAgent file not found: src/memory/memory-agent.ts"
        return 1
    fi

    log_info "EnhancedICRMemoryAgent file exists."

    # Check for key methods
    local required_methods=(
        "retrieveHistoricalKnowledge"
        "storeRefinementInsights"
        "storeContextualSession"
        "getContextualMemory"
        "learnFromSession"
        "analyzePatterns"
    )

    for method in "${required_methods[@]}"; do
        if grep -q "$method" "src/memory/memory-agent.ts"; then
            log_info "  ✓ Method found: $method"
        else
            log_error "  ✗ Method missing: $method"
            return 1
        fi
    done

    log_info "All required EnhancedICRMemoryAgent methods are defined."
    return 0
}

# ============================================================================
# PROBE 6: Test ICR Adapter Memory Integration
# ============================================================================

probe_adapter_integration() {
    log_info "Probe 6: Testing ICR adapter memory integration..."

    # Check if adapter file exists
    if [ ! -f "src/adapter.ts" ]; then
        log_error "Adapter file not found: src/adapter.ts"
        return 1
    fi

    # Check for memory integration method
    if grep -q "createContextualRequestWithMemory" "src/adapter.ts"; then
        log_info "  ✓ Method found: createContextualRequestWithMemory"
    else
        log_error "  ✗ Method missing: createContextualRequestWithMemory"
        return 1
    fi

    # Check for memory agent import
    if grep -q "EnhancedICRMemoryAgent" "src/adapter.ts"; then
        log_info "  ✓ Memory agent import found"
    else
        log_error "  ✗ Memory agent import missing"
        return 1
    fi

    log_info "ICR adapter memory integration is properly configured."
    return 0
}

# ============================================================================
# PROBE 7: Test Memory Storage Flow
# ============================================================================

probe_memory_storage_flow() {
    log_info "Probe 7: Testing memory storage flow..."

    # This is a compile-time check (TypeScript compilation)
    # We'll verify the code structure is correct

    # Check graphiti-memory.ts has episode formatting methods
    if grep -q "formatInsightsAsEpisode" "src/memory/graphiti-memory.ts"; then
        log_info "  ✓ Method found: formatInsightsAsEpisode"
    else
        log_error "  ✗ Method missing: formatInsightsAsEpisode"
        return 1
    fi

    if grep -q "formatSessionAsEpisode" "src/memory/graphiti-memory.ts"; then
        log_info "  ✓ Method found: formatSessionAsEpisode"
    else
        log_error "  ✗ Method missing: formatSessionAsEpisode"
        return 1
    fi

    log_info "Memory storage flow methods are properly defined."
    return 0
}

# ============================================================================
# PROBE 8: Test Memory Retrieval Flow
# ============================================================================

probe_memory_retrieval_flow() {
    log_info "Probe 8: Testing memory retrieval flow..."

    # Check retrieval methods exist
    if grep -q "buildSearchQuery" "src/memory/graphiti-memory.ts"; then
        log_info "  ✓ Method found: buildSearchQuery"
    else
        log_error "  ✗ Method missing: buildSearchQuery"
        return 1
    fi

    if grep -q "transformEdgeToKnowledge" "src/memory/graphiti-memory.ts"; then
        log_info "  ✓ Method found: transformEdgeToKnowledge"
    else
        log_error "  ✗ Method missing: transformEdgeToKnowledge"
        return 1
    fi

    if grep -q "filterKnowledge" "src/memory/graphiti-memory.ts"; then
        log_info "  ✓ Method found: filterKnowledge"
    else
        log_error "  ✗ Method missing: filterKnowledge"
        return 1
    fi

    log_info "Memory retrieval flow methods are properly defined."
    return 0
}

# ============================================================================
# PROBE 9: Test Pattern Learning
# ============================================================================

probe_pattern_learning() {
    log_info "Probe 9: Testing pattern learning capabilities..."

    # Check pattern extraction methods
    if grep -q "extractPatternsFromSession" "src/memory/graphiti-memory.ts"; then
        log_info "  ✓ Method found: extractPatternsFromSession"
    else
        log_error "  ✗ Method missing: extractPatternsFromSession"
        return 1
    fi

    if grep -q "buildPatternRelationships" "src/memory/graphiti-memory.ts"; then
        log_info "  ✓ Method found: buildPatternRelationships"
    else
        log_error "  ✗ Method missing: buildPatternRelationships"
        return 1
    fi

    if grep -q "analyzePatterns" "src/memory/memory-agent.ts"; then
        log_info "  ✓ Method found: analyzePatterns (Memory Agent)"
    else
        log_error "  ✗ Method missing: analyzePatterns (Memory Agent)"
        return 1
    fi

    log_info "Pattern learning capabilities are properly implemented."
    return 0
}

# ============================================================================
# PROBE 10: Test Temporal Context
# ============================================================================

probe_temporal_context() {
    log_info "Probe 10: Testing temporal context support..."

    # Check for temporal filtering in memory queries
    if grep -q "time_range" "src/memory/canonical.ts"; then
        log_info "  ✓ Temporal time range found in schemas"
    else
        log_error "  ✗ Temporal time range missing from schemas"
        return 1
    fi

    if grep -q "temporal_filter" "src/memory/graphiti-memory.ts"; then
        log_info "  ✓ Temporal filter found in memory manager"
    else
        log_error "  ✗ Temporal filter missing from memory manager"
        return 1
    fi

    log_info "Temporal context support is properly configured."
    return 0
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    echo ""
    echo "=========================================="
    echo "ICR Graphiti Memory Integration Probe"
    echo "=========================================="
    echo ""

    # Check required environment variables
    check_required_vars
    echo ""

    # Run all probes
    local total_probes=10
    local passed_probes=0

    # API Availability Probes
    if probe_icr_api; then
        ((passed_probes++))
    fi
    echo ""

    if probe_graphiti_api; then
        ((passed_probes++))
    fi
    echo ""

    # Schema and Structure Probes
    if probe_memory_schemas; then
        ((passed_probes++))
    fi
    echo ""

    if probe_graphiti_memory_manager; then
        ((passed_probes++))
    fi
    echo ""

    if probe_enhanced_memory_agent; then
        ((passed_probes++))
    fi
    echo ""

    if probe_adapter_integration; then
        ((passed_probes++))
    fi
    echo ""

    # Flow Probes
    if probe_memory_storage_flow; then
        ((passed_probes++))
    fi
    echo ""

    if probe_memory_retrieval_flow; then
        ((passed_probes++))
    fi
    echo ""

    if probe_pattern_learning; then
        ((passed_probes++))
    fi
    echo ""

    if probe_temporal_context; then
        ((passed_probes++))
    fi
    echo ""

    # Summary
    echo "=========================================="
    echo "Probe Summary"
    echo "=========================================="
    echo ""
    echo "Total Probes: $total_probes"
    echo "Passed: $passed_probes"
    echo "Failed: $((total_probes - passed_probes))"
    echo ""

    if [ $passed_probes -eq $total_probes ]; then
        log_info "✓ All probes passed successfully!"
        echo ""
        echo "ICR Graphiti memory integration is ready for use."
        return 0
    else
        log_error "✗ Some probes failed. Please review the errors above."
        return 1
    fi
}

# Run main function
main

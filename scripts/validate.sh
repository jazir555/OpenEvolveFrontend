#!/bin/bash
# =============================================================================
# OpenEvolve Validation Script
# License: Apache 2.0
# Description: Validates service health and configuration
# Usage: ./validate.sh [--report-only]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_FILE="${PROJECT_ROOT}/logs/validation-report-$(date +%Y%m%d-%H%M%S).md"
REPORT_ONLY=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[FAIL]${NC} $*"; }

print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --report-only)
            REPORT_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--report-only]"
            echo "  --report-only    Only generate report without console output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Start report
mkdir -p "$(dirname "${REPORT_FILE}")"
cat > "${REPORT_FILE}" << EOF
# OpenEvolve Validation Report
**Generated:** $(date '+%Y-%m-%d %H:%M:%S')

EOF

if [ "$REPORT_ONLY" = false ]; then
    print_header "STEP 1: Checking Services Status"
fi

echo "## Service Status" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"

cd "${PROJECT_ROOT}"
services_running=$(docker compose ps --services --filter "status=running" 2>/dev/null | wc -l)
services_total=$(docker compose ps --services 2>/dev/null | wc -l)

echo "| Service | Status | Health |" >> "${REPORT_FILE}"
echo "|---------|--------|--------|" >> "${REPORT_FILE}"

for service in $(docker compose ps --services 2>/dev/null); do
    status=$(docker compose ps -a "$service" --format "{{.Status}}" 2>/dev/null)
    health=$(docker compose ps -a "$service" --format "{{.Health}}" 2>/dev/null)
    echo "| $service | $status | ${health:-N/A} |" >> "${REPORT_FILE}"

    if [ "$REPORT_ONLY" = false ]; then
        if echo "$status" | grep -q "running"; then
            if [ "$health" = "healthy" ] || [ -z "$health" ]; then
                log_success "$service: $status"
            else
                log_warning "$service: $status (health: $health)"
            fi
        else
            log_error "$service: $status"
        fi
    fi
done

echo "" >> "${REPORT_FILE}"
echo "**Summary:** $services_running/$services_total services running" >> "${REPORT_FILE}"

if [ "$REPORT_ONLY" = false ]; then
    echo ""
fi

if [ "$REPORT_ONLY" = false ]; then
    print_header "STEP 2: Querying Health Endpoints"
fi

echo "## Health Checks" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"

declare -A endpoints=(
    ["OpenEvolve API"]="http://localhost:8000/health"
    ["Orchestrator"]="http://localhost:8080/health"
    ["Prometheus"]="http://localhost:9090/-/healthy"
)

echo "| Service | Endpoint | Status |" >> "${REPORT_FILE}"
echo "|---------|----------|--------|" >> "${REPORT_FILE}"

for service in "${!endpoints[@]}"; do
    url="${endpoints[$service]}"

    if response=$(curl -s -w "\n%{http_code}" --max-time 5 "$url" 2>/dev/null); then
        http_code=$(echo "$response" | tail -n1)

        if [ "$http_code" = "200" ] || [ "$http_code" = "204" ]; then
            if [ "$REPORT_ONLY" = false ]; then
                log_success "$service: HTTP $http_code"
            fi
            echo "| $service | $url | ✅ $http_code |" >> "${REPORT_FILE}"
        else
            if [ "$REPORT_ONLY" = false ]; then
                log_warning "$service: HTTP $http_code"
            fi
            echo "| $service | $url | ⚠️ $http_code |" >> "${REPORT_FILE}"
        fi
    else
        if [ "$REPORT_ONLY" = false ]; then
            log_error "$service: Connection failed"
        fi
        echo "| $service | $url | ❌ Failed |" >> "${REPORT_FILE}"
    fi
done

echo "" >> "${REPORT_FILE}"

if [ "$REPORT_ONLY" = false ]; then
    print_header "STEP 3: Checking Logs for Errors"
fi

echo "## Log Analysis" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"

if [ "$REPORT_ONLY" = false ]; then
    log_info "Analyzing recent logs for errors..."
fi

error_count=$(docker compose logs --since=5m 2>&1 | grep -i "error\|exception\|fatal" | wc -l)
warning_count=$(docker compose logs --since=5m 2>&1 | grep -i "warning" | wc -l)

echo "### Error Summary (Last 5 Minutes)" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"
echo "- **Errors:** $error_count" >> "${REPORT_FILE}"
echo "- **Warnings:** $warning_count" >> "${REPORT_FILE}"

if [ "$REPORT_ONLY" = false ]; then
    if [ $error_count -gt 0 ]; then
        log_warning "Found $error_count errors in recent logs"
    else
        log_success "No errors found in recent logs"
    fi
fi

echo "" >> "${REPORT_FILE}"

if [ "$REPORT_ONLY" = false ]; then
    print_header "STEP 4: Generating Validation Report"
fi

echo "## Summary" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"

# Calculate overall health
if [ $services_running -eq $services_total ] && [ $error_count -eq 0 ]; then
    overall_status="✅ Healthy"
    if [ "$REPORT_ONLY" = false ]; then
        log_success "Overall System Status: HEALTHY"
    fi
elif [ $services_running -gt 0 ]; then
    overall_status="⚠️ Degraded"
    if [ "$REPORT_ONLY" = false ]; then
        log_warning "Overall System Status: DEGRADED"
    fi
else
    overall_status="❌ Unhealthy"
    if [ "$REPORT_ONLY" = false ]; then
        log_error "Overall System Status: UNHEALTHY"
    fi
fi

echo "**Overall Status:** $overall_status" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"

if [ "$REPORT_ONLY" = false ]; then
    echo ""
    log_success "Validation complete. Report saved to: ${REPORT_FILE}"
    echo ""
fi

# Exit with appropriate code
if [ "$overall_status" = "✅ Healthy" ]; then
    exit 0
elif [ "$overall_status" = "⚠️ Degraded" ]; then
    exit 1
else
    exit 2
fi

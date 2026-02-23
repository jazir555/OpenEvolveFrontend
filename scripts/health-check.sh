#!/bin/bash
# =============================================================================
# OpenEvolve Health Check Script
# License: Apache 2.0
# Description: Check health status of all services
# Usage: ./health-check.sh [--json] [--quiet]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

OUTPUT_FORMAT="table"
VERBOSE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            OUTPUT_FORMAT="json"
            shift
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Check health status of all OpenEvolve services"
            echo ""
            echo "OPTIONS:"
            echo "  --json      Output in JSON format"
            echo "  --quiet     Suppress verbose output"
            echo "  -h, --help  Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "${PROJECT_ROOT}"

# Service health check configuration
declare -A SERVICES=(
    ["openevolve-app"]="http://localhost:8080/health"
    ["openevolve-valkey"]="tcp://localhost:6379"
    ["openevolve-prometheus"]="http://localhost:9090/-/healthy"
    ["openevolve-grafana"]="http://localhost:3000/api/health"
)

declare -A SERVICE_STATUS
declare -A SERVICE_RESPONSE

# Check each service
for service in "${!SERVICES[@]}"; do
    endpoint="${SERVICES[$service]}"

    # Check if container is running
    if ! docker compose ps -q "$service" &>/dev/null; then
        SERVICE_STATUS[$service]="stopped"
        SERVICE_RESPONSE[$service]="Container not running"
        continue
    fi

    # Check health endpoint
    if [[ $endpoint == http* ]]; then
        if response=$(curl -s -w "%{http_code}" --max-time 3 "$endpoint" 2>/dev/null); then
            http_code="${response: -3}"
            if [ "$http_code" = "200" ] || [ "$http_code" = "204" ]; then
                SERVICE_STATUS[$service]="healthy"
                SERVICE_RESPONSE[$service]="HTTP $http_code"
            else
                SERVICE_STATUS[$service]="unhealthy"
                SERVICE_RESPONSE[$service]="HTTP $http_code"
            fi
        else
            SERVICE_STATUS[$service]="unreachable"
            SERVICE_RESPONSE[$service]="Connection failed"
        fi
    elif [[ $endpoint == tcp* ]]; then
        host_port="${endpoint#tcp://}"
        if nc -z -w3 "${host_port%:*}" "${host_port#*:}" 2>/dev/null; then
            SERVICE_STATUS[$service]="healthy"
            SERVICE_RESPONSE[$service]="Port open"
        else
            SERVICE_STATUS[$service]="unreachable"
            SERVICE_RESPONSE[$service]="Port closed"
        fi
    fi
done

# Output results
if [ "$OUTPUT_FORMAT" = "json" ]; then
    echo "{"
    first=true
    for service in "${!SERVICE_STATUS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        printf '  "%s": {"status": "%s", "response": "%s"}' \
            "$service" "${SERVICE_STATUS[$service]}" "${SERVICE_RESPONSE[$service]}"
    done
    echo ""
    echo "}"
else
    # Table format
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}                    OpenEvolve Health Check                   ${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        echo ""
    fi

    printf "%-30s %-15s %-20s\n" "Service" "Status" "Response"
    printf "%-30s %-15s %-20s\n" "-------" "------" "--------"

    all_healthy=true
    for service in "${!SERVICE_STATUS[@]}"; do
        status="${SERVICE_STATUS[$service]}"
        response="${SERVICE_RESPONSE[$service]}"

        case $status in
            healthy)
                status_color="$GREEN"
                ;;
            unhealthy|unreachable)
                status_color="$RED"
                all_healthy=false
                ;;
            stopped)
                status_color="$YELLOW"
                all_healthy=false
                ;;
        esac

        printf "%-30s ${status_color}%-15s${NC} %-20s\n" "$service" "$status" "$response"
    done

    echo ""

    if [ "$VERBOSE" = true ]; then
        if [ "$all_healthy" = true ]; then
            echo -e "${GREEN}✓ All services are healthy${NC}"
        else
            echo -e "${RED}✗ Some services are not healthy${NC}"
        fi
    fi
fi

# Exit with appropriate code
if [ "$all_healthy" = true ]; then
    exit 0
else
    exit 1
fi

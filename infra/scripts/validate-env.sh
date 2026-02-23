#!/bin/bash
# Validate required environment variables for LoongFlow deployment
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: All config via env vars
# - Service must crash immediately if required vars are missing
#
# Usage:
#   source infra/scripts/validate-env.sh
#   ./infra/scripts/validate-env.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0

# Function to check required environment variables
check_required() {
  local var_name=$1
  if [ -z "${!1:-}" ]; then
    echo -e "${RED}❌ Required: $var_name is not set${NC}"
    ((ERRORS++))
    return 1
  else
    echo -e "${GREEN}✅ $var_name${NC}"
    return 0
  fi
}

# Function to check optional environment variables
check_optional() {
  local var_name=$1
  local default_value=$2
  if [ -z "${!1:-}" ]; then
    echo -e "${YELLOW}ℹ️  $var_name: (will use default: $default_value)${NC}"
    ((WARNINGS++))
    return 0
  else
    echo -e "${GREEN}✅ $var_name: ${!1}${NC}"
    return 0
  fi
}

# Function to validate numeric value
check_numeric() {
  local var_name=$1
  local value=$2
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}❌ $var_name must be numeric, got: $value${NC}"
    ((ERRORS++))
    return 1
  fi
  return 0
}

echo "================================================"
echo "LoongFlow Environment Validation"
echo "================================================"
echo ""

# ============================================================================
# Critical Variables (Service will crash if missing)
# ============================================================================
echo "Checking critical variables..."

# Core service required variables
check_required "LOONGFLOW_LLM_API_KEY"
check_required "LOONGFLOW_LLM_PROVIDER"

# Adapter required variables
check_required "LOONGFLOW_API_URL"

echo ""

# ============================================================================
# Optional Variables with Defaults
# ============================================================================
echo "Checking optional variables..."

# LLM configuration
check_optional "LOONGFLOW_LLM_MODEL" "gpt-4"
check_optional "LOONGFLOW_LLM_TEMPERATURE" "0.7"
check_optional "LOONGFLOW_LLM_MAX_TOKENS" "2000"

# Workflow configuration
check_optional "LOONGFLOW_MAX_CONCURRENT_WORKFLOWS" "10"
check_optional "LOONGFLOW_WORKFLOW_TIMEOUT_MS" "300000"

# Adapter configuration
check_optional "LOONGFLOW_TIMEOUT_MS" "30000"
check_optional "LOONGFLOW_MAX_RETRIES" "3"
check_optional "PORT" "8040"

# Logging
check_optional "LOG_LEVEL" "INFO"
check_optional "TZ" "UTC"

echo ""

# ============================================================================
# Validate Numeric Values
# ============================================================================
echo "Validating numeric values..."

if [ -n "${LOONGFLOW_TIMEOUT_MS:-}" ]; then
  check_numeric "LOONGFLOW_TIMEOUT_MS" "$LOONGFLOW_TIMEOUT_MS"
fi

if [ -n "${LOONGFLOW_MAX_RETRIES:-}" ]; then
  check_numeric "LOONGFLOW_MAX_RETRIES" "$LOONGFLOW_MAX_RETRIES"
fi

if [ -n "${LOONGFLOW_MAX_CONCURRENT_WORKFLOWS:-}" ]; then
  check_numeric "LOONGFLOW_MAX_CONCURRENT_WORKFLOWS" "$LOONGFLOW_MAX_CONCURRENT_WORKFLOWS"
fi

if [ -n "${LOONGFLOW_WORKFLOW_TIMEOUT_MS:-}" ]; then
  check_numeric "LOONGFLOW_WORKFLOW_TIMEOUT_MS" "$LOONGFLOW_WORKFLOW_TIMEOUT_MS"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "================================================"
if [ $ERRORS -gt 0 ]; then
  echo -e "${RED}❌ Validation failed with $ERRORS error(s)${NC}"
  echo ""
  echo "Required environment variables are missing."
  echo "Please set them before deploying:"
  echo ""
  echo "  export LOONGFLOW_LLM_API_KEY=sk-..."
  echo "  export LOONGFLOW_LLM_PROVIDER=openai"
  echo "  export LOONGFLOW_API_URL=http://loongflow-core:8000"
  echo ""
  exit 1
else
  echo -e "${GREEN}✅ Environment validation complete${NC}"
  echo -e "${YELLOW}ℹ️  $WARNINGS optional variable(s) will use defaults${NC}"
  echo ""
  echo "You can proceed with deployment."
  exit 0
fi

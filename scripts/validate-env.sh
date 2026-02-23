#!/bin/bash
# Validate required environment variables
# Following CLAUDE.md - Law of Configuration Explicitness

set -e

ENV_FILE=".env"

echo "================================================"
echo "OpenEvolve Environment Validation"
echo "================================================"
echo ""

if [ ! -f "$ENV_FILE" ]; then
  echo "❌ .env file not found"
  echo ""
  echo "Run: ./scripts/setup-env.sh"
  exit 1
fi

echo "🔍 Validating environment variables..."
echo ""

# Source the file to get variables
set -a
source "$ENV_FILE"
set +a

# Track validation results
REQUIRED_ERRORS=0
OPTIONAL_WARNINGS=0
SUCCESS_COUNT=0

# Function to check required variable
check_required() {
  local var_name=$1
  local description=$2

  if [ -z "${!var_name}" ]; then
    echo "❌ REQUIRED: $var_name"
    echo "   $description"
    REQUIRED_ERRORS=$((REQUIRED_ERRORS + 1))
  else
    # Check if it's still a placeholder
    local value="${!var_name}"
    if [[ "$value" == *"your-"* ]] || [[ "$value" == *"changeme"* ]] || [[ "$value" == *"here"* ]]; then
      echo "⚠️  PLACEHOLDER: $var_name"
      echo "   $description"
      echo "   Current value: $value"
      OPTIONAL_WARNINGS=$((OPTIONAL_WARNINGS + 1))
    else
      echo "✅ $var_name"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
  fi
}

# Function to check optional variable
check_optional() {
  local var_name=$1
  local default_value=$2

  if [ -z "${!var_name}" ]; then
    echo "ℹ️  OPTIONAL (not set): $var_name"
    echo "   Will use default: $default_value"
  else
    echo "✅ $var_name = ${!var_name}"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  fi
}

# Check infrastructure variables
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Infrastructure Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_required "EVENT_BUS_URL" "Event bus connection URL"
check_required "TZ" "Timezone (should be UTC)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "API Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_required "LOONGFLOW_API_URL" "LoongFlow Core API endpoint"
check_required "OPENEVOLVE_API_URL" "OpenEvolve API endpoint"
check_required "BUBBLELAB_API_URL" "BubbleLab API endpoint"
check_required "RAGBITS_API_URL" "RagBits API endpoint"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Service Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_optional "TIMEOUT_MS" "30000"
check_optional "MAX_RETRIES" "3"
check_optional "LOG_LEVEL" "INFO"
check_optional "LOG_FORMAT" "json"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "LLM Provider Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_required "LOONGFLOW_LLM_API_KEY" "OpenAI API key for LoongFlow"
check_optional "OPENAI_API_KEY" "sk-..."
check_optional "ANTHROPIC_API_KEY" "sk-ant-..."
check_optional "GOOGLE_API_KEY" "..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Database Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_required "NEO4J_URI" "Neo4j connection URI"
check_required "NEO4J_USER" "Neo4j username"
check_required "NEO4J_PASSWORD" "Neo4j password"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Development Settings"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_optional "DEBUG" "false"
check_optional "SKIP_CONTRACT_TESTS" "false"

# Validation Summary
echo ""
echo "================================================"
echo "Validation Summary"
echo "================================================"
echo "✅ Valid variables: $SUCCESS_COUNT"
echo "⚠️  Placeholders: $OPTIONAL_WARNINGS"
echo "❌ Missing required: $REQUIRED_ERRORS"
echo ""

if [ $REQUIRED_ERRORS -gt 0 ]; then
  echo "❌ VALIDATION FAILED"
  echo ""
  echo "Required environment variables are missing!"
  echo ""
  echo "Please set the required variables in .env before starting services."
  echo ""
  echo "Edit .env:"
  echo "  nano .env"
  echo "  or"
  echo "  code .env"
  echo ""
  exit 1
elif [ $OPTIONAL_WARNINGS -gt 0 ]; then
  echo "⚠️  VALIDATION PASSED WITH WARNINGS"
  echo ""
  echo "Some variables contain placeholder values."
  echo "Services may not start correctly without proper values."
  echo ""
  echo "Please review and update placeholder values in .env"
  exit 0
else
  echo "✅ VALIDATION PASSED"
  echo ""
  echo "All required environment variables are set!"
  echo ""
  echo "You can now start services:"
  echo ""
  echo "1. Start infrastructure:"
  echo "   docker-compose -f docker-compose.infrastructure.yml up -d"
  echo ""
  echo "2. Start adapters:"
  echo "   docker-compose -f infra/docker-compose-all-adapters.yml up -d"
  echo ""
  echo "3. Check health:"
  echo "   ./scripts/health-check.sh"
  echo ""
  exit 0
fi

#!/bin/bash
# Set up environment variables for Docker Compose
# Following CLAUDE.md - Law of Configuration Explicitness

set -e

ENV_FILE=".env"
INFRA_EXAMPLE="infra/.env.example"
LOONGFLOW_EXAMPLE="infra/.env.loongflow.example"

echo "================================================"
echo "OpenEvolve Environment Setup"
echo "================================================"
echo ""

# Check if .env exists
if [ -f "$ENV_FILE" ]; then
  echo "⚠️  .env file already exists"
  read -p "Overwrite? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting..."
    exit 0
  fi
  echo "Backing up existing .env to .env.backup"
  cp "$ENV_FILE" ".env.backup"
fi

# Determine which environment to set up
echo ""
echo "Which environment do you want to set up?"
echo "1) Full OpenEvolve Federation (all adapters)"
echo "2) LoongFlow Core only"
echo "3) Minimal development setup"
echo ""
read -p "Choose (1-3): " -n 1 -r
echo ""

case $REPLY in
  1)
    echo "📋 Setting up Full OpenEvolve Federation environment..."
    if [ -f "$INFRA_EXAMPLE" ]; then
      cp "$INFRA_EXAMPLE" "$ENV_FILE"
      echo "✅ Copied from $INFRA_EXAMPLE"
    else
      echo "⚠️  Example file not found, creating minimal .env"
      create_minimal_env
    fi
    ;;
  2)
    echo "📋 Setting up LoongFlow Core environment..."
    if [ -f "$LOONGFLOW_EXAMPLE" ]; then
      cp "$LOONGFLOW_EXAMPLE" "$ENV_FILE"
      echo "✅ Copied from $LOONGFLOW_EXAMPLE"
    else
      echo "⚠️  Example file not found, creating minimal .env"
      create_minimal_env
    fi
    ;;
  3)
    echo "📝 Creating minimal development environment..."
    create_minimal_env
    ;;
  *)
    echo "❌ Invalid choice"
    exit 1
    ;;
esac

echo ""
echo "✅ Environment file created: $ENV_FILE"
echo ""
echo "⚠️  IMPORTANT: Review .env and update values before starting services!"
echo ""
echo "Required variables to set:"
echo "  - API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)"
echo "  - Service URLs (if not using defaults)"
echo "  - Database credentials (NEO4J_PASSWORD, etc.)"
echo ""
echo "Next steps:"
echo "1. Edit .env with your values:"
echo "   nano .env"
echo "   or"
echo "   code .env"
echo ""
echo "2. Start infrastructure:"
echo "   docker-compose -f docker-compose.infrastructure.yml up -d"
echo ""
echo "3. Start services:"
echo "   docker-compose -f infra/docker-compose-all-adapters.yml up -d"
echo ""

# Function to create minimal environment
create_minimal_env() {
  cat > "$ENV_FILE" <<EOF
# OpenEvolve Environment Configuration
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
#
# Following CLAUDE.md - Law of Configuration Explicitness:
# - All values must be explicitly configured
# - NO magic defaults - services will crash if required values are missing
# - All timestamps in UTC (Law of UTC)

# =============================================================================
# Infrastructure Configuration
# =============================================================================

# Event Bus (Redis)
EVENT_BUS_URL=redis://event-bus:6379
REDIS_PORT=6379

# Orchestrator
ORCHESTRATOR_PORT=8080

# =============================================================================
# Logging Configuration
# =============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Log format: json or text
LOG_FORMAT=json

# =============================================================================
# Timezone (Law of UTC)
# =============================================================================

# All services MUST use UTC (Law of UTC)
TZ=UTC

# =============================================================================
# API Endpoints (Docker Service Names)
# =============================================================================

# Core Project APIs (internal Docker service names)
LOONGFLOW_API_URL=http://loongflow-core:8050
OPENEVOLVE_API_URL=http://openevolve-core:8000
BUBBLELAB_API_URL=http://bubblelab-core:8501
RAGBITS_API_URL=http://ragbits-core:8000

# =============================================================================
# Service Configuration
# =============================================================================

# Timeouts (milliseconds)
TIMEOUT_MS=30000
LOONGFLOW_TIMEOUT_MS=30000
OPENEVOLVE_TIMEOUT_MS=30000
BUBBLELAB_TIMEOUT_MS=30000

# Retries
MAX_RETRIES=3

# =============================================================================
# LLM Provider Configuration (REQUIRED)
# =============================================================================

# OpenAI API Key (REQUIRED for LoongFlow)
# LOONGFLOW_LLM_API_KEY=sk-your-openai-api-key-here
# OPENAI_API_KEY=sk-your-openai-api-key-here

# Anthropic API Key (optional)
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Google API Key (optional)
# GOOGLE_API_KEY=your-google-api-key-here

# =============================================================================
# Database Configuration
# =============================================================================

# Neo4j Configuration (for Graphiti)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password-here

# =============================================================================
# Contract Testing
# =============================================================================

# Skip contract tests on startup (useful for development)
SKIP_CONTRACT_TESTS=false

# =============================================================================
# Development Settings
# =============================================================================

# Enable debug mode (set to false in production)
DEBUG=false

EOF
}

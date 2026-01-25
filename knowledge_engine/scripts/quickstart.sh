#!/bin/bash

# Neo4j Quick Start Script
# OpenEvolve Knowledge Engine - Phase 1.1.1
#
# This script automates the setup and initialization of Neo4j
# Usage: ./quickstart.sh [dev|prod]

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

ENVIRONMENT="${1:-dev}"

echo -e "${BLUE}"
echo "================================================"
echo "Neo4j Quick Start - OpenEvolve Knowledge Engine"
echo "================================================"
echo -e "${NC}"
echo ""

# Validate environment choice
if [ "$ENVIRONMENT" != "dev" ] && [ "$ENVIRONMENT" != "prod" ]; then
    echo "Usage: $0 [dev|prod]"
    exit 1
fi

# Set environment file
if [ "$ENVIRONMENT" = "dev" ]; then
    ENV_FILE="knowledge_engine/config/neo4j.dev.env"
    echo -e "${GREEN}Starting Neo4j in DEVELOPMENT mode...${NC}"
else
    ENV_FILE="knowledge_engine/config/neo4j.prod.env"
    echo -e "${GREEN}Starting Neo4j in PRODUCTION mode...${NC}"
fi

echo ""

# ============================================================================
# STEP 1: Create necessary directories
# ============================================================================
echo -e "${YELLOW}[1/6] Creating data directories...${NC}"

mkdir -p data/neo4j/{data,logs,import,plugins,backups}
chmod -R 777 data/neo4j

echo -e "${GREEN}✓ Data directories created${NC}"
echo ""

# ============================================================================
# STEP 2: Stop any existing Neo4j container
# ============================================================================
echo -e "${YELLOW}[2/6] Checking for existing containers...${NC}"

if docker ps -a | grep -q openevolve-neo4j; then
    echo "Stopping existing Neo4j container..."
    docker stop openevolve-neo4j 2>/dev/null || true
    docker rm openevolve-neo4j 2>/dev/null || true
    echo -e "${GREEN}✓ Existing container removed${NC}"
else
    echo -e "${GREEN}✓ No existing container found${NC}"
fi

echo ""

# ============================================================================
# STEP 3: Start Neo4j
# ============================================================================
echo -e "${YELLOW}[3/6] Starting Neo4j container...${NC}"

docker-compose -f docker-compose.neo4j.yml --env-file "$ENV_FILE" up -d neo4j

echo -e "${GREEN}✓ Neo4j container started${NC}"
echo ""

# ============================================================================
# STEP 4: Wait for Neo4j to be healthy
# ============================================================================
echo -e "${YELLOW}[4/6] Waiting for Neo4j to be ready...${NC}"

MAX_WAIT=60
WAIT_TIME=0
HEALTH_CHECK_INTERVAL=2

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if docker exec openevolve-neo4j wget -O /dev/null http://localhost:7474 2>/dev/null; then
        echo -e "${GREEN}✓ Neo4j is ready!${NC}"
        break
    fi

    echo -n "."
    sleep $HEALTH_CHECK_INTERVAL
    WAIT_TIME=$((WAIT_TIME + HEALTH_CHECK_INTERVAL))
done

echo ""

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo -e "\033[0;31m✗ Neo4j failed to start within ${MAX_WAIT} seconds\033[0m"
    echo "Check logs with: docker logs openevolve-neo4j"
    exit 1
fi

echo ""

# ============================================================================
# STEP 5: Initialize database
# ============================================================================
echo -e "${YELLOW}[5/6] Initializing database...${NC}"

# Run initialization script
docker exec -it openevolve-neo4j cypher-shell \
    -u neo4j \
    -p openevolve2026 \
    -f /scripts/init_neo4j.cypher

echo -e "${GREEN}✓ Database initialized${NC}"
echo ""

# ============================================================================
# STEP 6: Run health check
# ============================================================================
echo -e "${YELLOW}[6/6] Running health check...${NC}"

sleep 5  # Give Neo4j a moment to settle

# Run health check script
bash knowledge_engine/scripts/health_check.sh

echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "${GREEN}"
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo -e "${NC}"
echo ""
echo "Neo4j is now running and ready to use!"
echo ""
echo "Connection Details:"
echo "  - Bolt URI:       bolt://localhost:7687"
echo "  - HTTP URI:       http://localhost:7474"
echo "  - Username:       neo4j"
echo "  - Password:       openevolve2026"
echo ""
echo "Quick Commands:"
echo "  - Open Neo4j Browser:  open http://localhost:7474"
echo "  - Run health check:    bash knowledge_engine/scripts/health_check.sh"
echo "  - Create backup:       bash knowledge_engine/scripts/backup.sh"
echo "  - View logs:           docker logs -f openevolve-neo4j"
echo "  - Stop Neo4j:          docker stop openevolve-neo4j"
echo ""
echo "For more information, see: knowledge_engine/docs/neo4j_setup.md"
echo ""

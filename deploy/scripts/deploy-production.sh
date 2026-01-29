#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Deploying OpenEvolve to PRODUCTION..."
echo ""

# Safety confirmation
read -p "Are you sure you want to deploy to PRODUCTION? (yes/no): " confirmation
if [ "$confirmation" != "yes" ]; then
    echo "Deployment aborted."
    exit 0
fi

# Load environment variables
if [ ! -f .env.production ]; then
    echo -e "${RED}ERROR: .env.production file not found${NC}"
    exit 1
fi

set -a
source .env.production
set +a

# Validate required environment variables
required_vars=("DATABASE_URL" "REDIS_URL" "JWT_SECRET" "DB_USER" "DB_PASSWORD" "DB_NAME")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}ERROR: $var is not set in .env.production${NC}"
        exit 1
    fi
done

# Pre-deployment checks
echo -e "${YELLOW}🔍 Running pre-deployment checks...${NC}"
bash ../scripts/pre-deploy-checks.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Pre-deployment checks failed${NC}"
    exit 1
fi

# Backup current deployment
echo -e "${YELLOW}💾 Backing up current deployment...${NC}"
bash ../scripts/backup-production.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Backup failed${NC}"
    exit 1
fi

# Navigate to production directory
cd "$(dirname "$0")/../production"

# Pull latest images
echo -e "${YELLOW}📦 Pulling latest images...${NC}"
docker-compose -f docker-compose.production.yml pull

# Build new images
echo -e "${YELLOW}🔨 Building Docker images...${NC}"
docker-compose -f docker-compose.production.yml build --no-cache

# Deploy with zero-downtime (rolling update)
echo -e "${YELLOW}📦 Deploying new version...${NC}"
IMAGE_TAG=$(git rev-parse --short HEAD)
export IMAGE_TAG

# Start new containers alongside existing ones
echo -e "${YELLOW}🔄 Starting new containers...${NC}"
docker-compose -f docker-compose.production.yml up -d --no-deps --build

# Wait for new containers to be healthy
echo -e "${YELLOW}⏳ Waiting for containers to be healthy...${NC}"
sleep 60

# Rolling update for API Gateway
echo -e "${YELLOW}🔄 Executing rolling update...${NC}"
docker-compose -f docker-compose.production.yml up -d --scale api-gateway=3 --no-recreate

# Post-deployment checks
echo -e "${YELLOW}🔍 Running post-deployment checks...${NC}"
bash ../scripts/post-deploy-checks.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Post-deployment checks failed${NC}"
    echo "Initiating rollback..."
    bash ../scripts/rollback.sh
    exit 1
fi

# Run smoke tests
echo -e "${YELLOW}💨 Running smoke tests...${NC}"
bash ../scripts/smoke-tests.sh production https://openevolve.ai
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Smoke tests failed${NC}"
    echo "Initiating rollback..."
    bash ../scripts/rollback.sh
    exit 1
fi

echo -e "${GREEN}✅ Production deployment complete!${NC}"
echo -e "${GREEN}📍 Production URL: https://openevolve.ai${NC}"
echo ""
echo -e "${YELLOW}📊 Container Status:${NC}"
docker-compose -f docker-compose.production.yml ps

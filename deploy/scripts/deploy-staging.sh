#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Deploying OpenEvolve to Staging..."
echo ""

# Load environment variables
if [ ! -f .env.staging ]; then
    echo -e "${RED}ERROR: .env.staging file not found${NC}"
    echo "Please create .env.staging with required environment variables"
    exit 1
fi

set -a
source .env.staging
set +a

# Validate required environment variables
required_vars=("DB_PASSWORD" "JWT_SECRET")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}ERROR: $var is not set in .env.staging${NC}"
        exit 1
    fi
done

# Pre-deployment checks
echo -e "${YELLOW}🔍 Running pre-deployment checks...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose is not installed${NC}"
    exit 1
fi

# Navigate to staging directory
cd "$(dirname "$0")/../staging"

# Pull latest images
echo -e "${YELLOW}📦 Pulling latest images...${NC}"
docker-compose -f docker-compose.staging.yml pull

# Build images
echo -e "${YELLOW}🔨 Building Docker images...${NC}"
docker-compose -f docker-compose.staging.yml build --no-cache

# Stop existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose.staging.yml down -v

# Start new containers
echo -e "${YELLOW}🚀 Starting new containers...${NC}"
docker-compose -f docker-compose.staging.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 30

# Run health checks
echo -e "${YELLOW}🏥 Running health checks...${NC}"
bash ../scripts/health-check.sh staging http://localhost:8000

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Staging deployment complete!${NC}"
    echo -e "${GREEN}📍 Staging URL: http://staging.openevolve.ai${NC}"
    echo -e "${GREEN}📍 API URL: http://localhost:8000${NC}"
    echo -e "${GREEN}📍 Frontend URL: http://localhost:3000${NC}"

    # Show container status
    echo ""
    echo -e "${YELLOW}📊 Container Status:${NC}"
    docker-compose -f docker-compose.staging.yml ps
else
    echo -e "${RED}❌ Health checks failed!${NC}"
    echo "Check logs with: docker-compose -f deploy/staging/docker-compose.staging.yml logs"
    exit 1
fi

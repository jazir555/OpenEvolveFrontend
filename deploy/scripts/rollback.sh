#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔄 ROLLING BACK PRODUCTION DEPLOYMENT"
echo "====================================="
echo ""

# Safety confirmation
read -p "Are you sure you want to ROLLBACK production? (yes/no): " confirmation
if [ "$confirmation" != "yes" ]; then
    echo "Rollback aborted."
    exit 0
fi

# Find latest backup
echo "🔍 Finding latest backup..."
cd ../backups
latest_backup=$(ls -td production_* 2>/dev/null | head -1)

if [ -z "$latest_backup" ]; then
    echo -e "${RED}ERROR: No backups found${NC}"
    exit 1
fi

echo -e "${YELLOW}Found backup: $latest_backup${NC}"
read -p "Use this backup? (yes/no): " backup_confirmation

if [ "$backup_confirmation" != "yes" ]; then
    echo "Rollback aborted."
    exit 0
fi

BACKUP_DIR="../backups/$latest_backup"

echo ""
echo "🔄 Starting rollback process..."
echo ""

# Step 1: Stop current containers
echo -n "Step 1: Stopping current containers... "
cd ../production
docker-compose -f docker-compose.production.yml down
echo -e "${GREEN}✅ DONE${NC}"

# Step 2: Restore database
echo -n "Step 2: Restoring database... "
if docker-compose -f docker-compose.production.yml up -d postgres; then
    sleep 10  # Wait for postgres to start
    if docker exec -i openevolve-db-prod psql -U openvolve openvolve < "$BACKUP_DIR/database.sql"; then
        echo -e "${GREEN}✅ DONE${NC}"
    else
        echo -e "${RED}❌ FAILED${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ FAILED${NC}"
    exit 1
fi

# Step 3: Restore Docker volumes
echo -n "Step 3: Restoring Docker volumes... "
if docker run --rm \
  -v openevolve_production_postgres_data:/data \
  -v "$BACKUP_DIR:/backup" \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/postgres_volume.tar.gz -C /data"; then
    echo -e "${GREEN}✅ DONE${NC}"
else
    echo -e "${RED}❌ FAILED${NC}"
    exit 1
fi

# Step 4: Restore configuration
echo -n "Step 4: Restoring configuration files... "
cp "$BACKUP_DIR/config/docker-compose.production.yml" ../production/docker-compose.production.yml
cp "$BACKUP_DIR/config/nginx.conf" ../production/nginx.conf
cp "$BACKUP_DIR/config/.env.production" ../production/.env.production
echo -e "${GREEN}✅ DONE${NC}"

# Step 5: Restore SSL certificates
echo -n "Step 5: Restoring SSL certificates... "
if [ -d "$BACKUP_DIR/ssl" ]; then
    rm -rf ../production/ssl
    cp -r "$BACKUP_DIR/ssl" ../production/
    echo -e "${GREEN}✅ DONE${NC}"
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (no SSL certificates in backup)"
fi

# Step 6: Restore previous Docker images
echo -n "Step 6: Restoring Docker images... "
if [ -f "$BACKUP_DIR/docker_images.txt" ]; then
    while read -r line; do
        image=$(echo "$line" | awk '{print $1}')
        echo "  Pulling $image..."
        docker pull "$image" || echo "  Warning: Failed to pull $image"
    done < "$BACKUP_DIR/docker_images.txt"
    echo -e "${GREEN}✅ DONE${NC}"
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (no image info in backup)"
fi

# Step 7: Start containers
echo -n "Step 7: Starting containers... "
docker-compose -f docker-compose.production.yml up -d
echo -e "${GREEN}✅ DONE${NC}"

# Step 8: Wait for containers to be healthy
echo "Step 8: Waiting for containers to be healthy..."
sleep 30

# Step 9: Verify rollback
echo -n "Step 9: Verifying rollback... "
if curl -f -s https://openevolve.ai/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ SUCCESS${NC}"
else
    echo -e "${RED}❌ FAILED${NC} (API not responding)"
    echo "Manual intervention required!"
    exit 1
fi

# Step 10: Run health checks
echo "Step 10: Running health checks..."
bash ../scripts/health-check.sh production https://openevolve.ai

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ ROLLBACK COMPLETE${NC}"
    echo "📍 System restored to backup: $latest_backup"
    echo ""
    echo "📊 Next steps:"
    echo "  1. Monitor system logs: docker-compose -f deploy/production/docker-compose.production.yml logs -f"
    echo "  2. Verify user traffic"
    echo "  3. Investigate root cause of failure"
    echo "  4. Fix issues and schedule new deployment"
    exit 0
else
    echo ""
    echo -e "${RED}❌ ROLLBACK VERIFICATION FAILED${NC}"
    echo "Manual intervention required!"
    exit 1
fi

#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "💾 Backing up Production Deployment..."
echo ""

# Create backup directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="../backups/production_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

echo "📁 Backup directory: $BACKUP_DIR"

# Backup 1: Database
echo -n "🗄️  Backing up database... "
if docker exec openevolve-db-prod pg_dump -U openvolve openvolve > "$BACKUP_DIR/database.sql"; then
    echo -e "${GREEN}✅ SUCCESS${NC}"
else
    echo -e "${RED}❌ FAILED${NC}"
    exit 1
fi

# Backup 2: Docker volumes
echo -n "📦 Backing up Docker volumes... "
if docker run --rm \
  -v openevolve_production_postgres_data:/data \
  -v "$BACKUP_DIR:/backup" \
  alpine tar czf /backup/postgres_volume.tar.gz -C /data .; then
    echo -e "${GREEN}✅ SUCCESS${NC}"
else
    echo -e "${RED}❌ FAILED${NC}"
    exit 1
fi

# Backup 3: Configuration files
echo -n "📝 Backing up configuration files... "
mkdir -p "$BACKUP_DIR/config"
cp -r ../production/docker-compose.production.yml "$BACKUP_DIR/config/"
cp -r ../production/nginx.conf "$BACKUP_DIR/config/"
cp -r ../production/.env.production "$BACKUP_DIR/config/"
echo -e "${GREEN}✅ SUCCESS${NC}"

# Backup 4: SSL certificates
echo -n "🔐 Backing up SSL certificates... "
if [ -d "../production/ssl" ]; then
    cp -r ../production/ssl "$BACKUP_DIR/"
    echo -e "${GREEN}✅ SUCCESS${NC}"
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (SSL directory not found)"
fi

# Backup 5: Current deployment info
echo -n "📊 Recording deployment info... "
{
    echo "Backup Timestamp: $TIMESTAMP"
    echo "Git Commit: $(git rev-parse HEAD)"
    echo "Git Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Docker Images:"
    docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedAt}}" | grep openevolve
} > "$BACKUP_DIR/deployment_info.txt"
echo -e "${GREEN}✅ SUCCESS${NC}"

# Backup 6: Docker image tags
echo -n "🐳 Backing up Docker image info... "
docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | grep openevolve > "$BACKUP_DIR/docker_images.txt"
echo -e "${GREEN}✅ SUCCESS${NC}"

# Create backup manifest
echo -n "📋 Creating backup manifest... "
{
    echo "Backup Manifest"
    echo "==============="
    echo "Timestamp: $TIMESTAMP"
    echo "Location: $BACKUP_DIR"
    echo ""
    echo "Contents:"
    du -sh "$BACKUP_DIR"/* | awk '{print "  -", $2, "("$1")"}'
    echo ""
    echo "Total Size:"
    du -sh "$BACKUP_DIR" | awk '{print "  ", $1}'
} > "$BACKUP_DIR/MANIFEST.txt"
echo -e "${GREEN}✅ SUCCESS${NC}"

# Cleanup old backups (keep last 5)
echo -n "🧹 Cleaning up old backups... "
cd ../backups
backup_count=$(ls -d production_* 2>/dev/null | wc -l)
if [ "$backup_count" -gt 5 ]; then
    ls -td production_* | tail -n +6 | xargs rm -rf
    echo -e "${GREEN}✅ SUCCESS${NC} (removed $((backup_count - 5)) old backups)"
else
    echo -e "${GREEN}✅ SUCCESS${NC} (no cleanup needed, $backup_count backups present)"
fi

echo ""
echo -e "${GREEN}✅ BACKUP COMPLETE${NC}"
echo "📍 Backup location: $BACKUP_DIR"
echo "📊 Backup size: $(du -sh "$BACKUP_DIR" | awk '{print $1}')"

exit 0

#!/bin/bash
# OpenEvolve Deployment Setup Script
# This script prepares the environment for deployment

set -e

echo "🚀 OpenEvolve Deployment Setup"
echo "=============================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker is installed"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi
print_success "Docker Compose is installed"

# Check OpenSSL
if ! command -v openssl &> /dev/null; then
    print_error "OpenSSL is not installed"
    echo "Please install OpenSSL"
    exit 1
fi
print_success "OpenSSL is installed"

# Check available disk space (minimum 10GB)
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (minimum 10GB recommended)"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."

mkdir -p deploy/staging/ssl
mkdir -p deploy/staging/logs
mkdir -p deploy/production/ssl
mkdir -p deploy/production/logs
mkdir -p deploy/monitoring/grafana/dashboards
mkdir -p deploy/monitoring/grafana/provisioning
mkdir -p backups
mkdir -p logs

print_success "Directories created"

# Generate self-signed SSL for staging (if not exists)
echo ""
echo "🔐 Generating SSL certificates..."

if [ ! -f deploy/staging/ssl/cert.pem ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout deploy/staging/ssl/key.pem \
        -out deploy/staging/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=OpenEvolve/CN=staging.openevolve.ai" \
        2>/dev/null
    print_success "Staging SSL certificate generated"
else
    print_success "Staging SSL certificate already exists"
fi

if [ ! -f deploy/production/ssl/cert.pem ]; then
    print_warning "Production SSL certificate not found"
    echo "For production, use Let's Encrypt or provide your own certificates"
    echo "Run: certbot certonly --nginx -d openevolve.ai"
fi

# Create environment file templates
echo ""
echo "📝 Creating environment file templates..."

# Staging environment template
cat > deploy/staging/.env.staging.example << 'EOF'
# OpenEvolve Staging Environment
# IMPORTANT: Copy this file to .env.staging and fill in the values

# Database
DB_PASSWORD=CHANGE_THIS_SECURE_PASSWORD

# Security
JWT_SECRET=CHANGE_THIS_SECRET_MIN_32_CHARS

# Frontend
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000

# Monitoring
GRAFANA_ADMIN_PASSWORD=CHANGE_THIS_GRAFANA_PASSWORD
EOF

# Production environment template
cat > deploy/production/.env.production.example << 'EOF'
# OpenEvolve Production Environment
# IMPORTANT: NEVER commit this file to version control

# External Database (Use managed service)
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# External Redis (Use managed service)
REDIS_URL=redis://host:6379/0

# Security
JWT_SECRET=CHANGE_THIS_TO_SECURE_PRODUCTION_SECRET

# Monitoring
GRAFANA_ADMIN_PASSWORD=CHANGE_THIS_GRAFANA_PASSWORD
EOF

print_success "Environment file templates created"

# Set script permissions
echo ""
echo "🔧 Setting script permissions..."
chmod +x deploy/scripts/*.sh
print_success "Script permissions set"

# Create .gitignore for sensitive files
echo ""
echo "🔒 Securing sensitive files..."

cat >> .gitignore << 'EOF'

# Deployment secrets
deploy/staging/.env.staging
deploy/production/.env.production
deploy/staging/ssl/*.pem
deploy/production/ssl/*.pem
deploy/**/logs/
backups/
*.sql
EOF

print_success "Git ignore rules updated"

# Generate random passwords for examples
echo ""
echo "🔑 Generating example passwords..."

RANDOM_DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
RANDOM_JWT_SECRET=$(openssl rand -base64 42 | tr -d "=+/" | cut -c1-42)
RANDOM_GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)

echo ""
echo "Generated secure passwords (save these somewhere safe!):"
echo ""
echo "Staging Database Password: $RANDOM_DB_PASSWORD"
echo "JWT Secret: $RANDOM_JWT_SECRET"
echo "Grafana Password: $RANDOM_GRAFANA_PASSWORD"
echo ""

# Verify Docker is running
echo ""
echo "🐳 Verifying Docker is running..."
if docker info &> /dev/null; then
    print_success "Docker is running"
else
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_success "Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure staging environment:"
echo "   cp deploy/staging/.env.staging.example deploy/staging/.env.staging"
echo "   nano deploy/staging/.env.staging"
echo ""
echo "2. Deploy to staging:"
echo "   bash deploy/scripts/deploy-staging.sh"
echo ""
echo "3. Verify deployment:"
echo "   bash deploy/scripts/health-check.sh staging"
echo "   bash deploy/scripts/smoke-tests.sh staging"
echo ""
echo "4. For production, repeat steps 1-3 with production config"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
#!/bin/bash
# LoongFlow Quick Deployment Script
#
# Following CLAUDE.md principles:
# - Law of Configuration Explicitness: Verify all env vars before starting
# - Law of Runtime Truth: Health checks validate actual deployment
#
# Usage:
#   ./scripts/deploy-loongflow.sh [environment]
#   ./scripts/deploy-loongflow.sh local
#   ./scripts/deploy-loongflow.sh kubernetes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ENVIRONMENT=${1:-local}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  LoongFlow Quick Deployment${NC}"
echo -e "${BLUE}  Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if .env.loongflow exists
ENV_FILE="$PROJECT_ROOT/infra/.env.loongflow"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}⚠️  Environment file not found${NC}"
    echo "Creating from template..."
    cp "$PROJECT_ROOT/infra/.env.loongflow.example" "$ENV_FILE"
    echo -e "${RED}❌ Please edit $ENV_FILE with your configuration${NC}"
    echo "Required variables:"
    echo "  - LOONGFLOW_LLM_API_KEY"
    echo ""
    exit 1
fi

# Load environment variables
set -a
source "$ENV_FILE"
set +a

# Validate required variables
echo -e "${BLUE}Validating configuration...${NC}"

REQUIRED_VARS=("LOONGFLOW_LLM_API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please set these in $ENV_FILE"
    exit 1
fi

echo -e "${GREEN}✅ Configuration validated${NC}"
echo ""

# ============================================================================
# Local Deployment (Docker Compose)
# ============================================================================

if [ "$ENVIRONMENT" = "local" ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Deploying to Local (Docker Compose)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Create network if it doesn't exist
    echo "Creating federation-network..."
    docker network create federation-network 2>/dev/null || true
    echo -e "${GREEN}✅ Network ready${NC}"
    echo ""

    # Start LoongFlow Core
    echo "Starting LoongFlow Core..."
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.loongflow-core.yml --env-file "$ENV_FILE" up -d

    echo "Waiting for Core to be healthy..."
    for i in {1..30}; do
        if curl -s http://localhost:8050/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Core is healthy${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ Core failed to start${NC}"
            echo "Check logs: docker-compose -f docker-compose.loongflow-core.yml logs"
            exit 1
        fi
        sleep 2
    done
    echo ""

    # Start LoongFlow Adapter
    echo "Starting LoongFlow Adapter..."
    docker-compose -f infra/docker-compose-all-adapters.yml --env-file "$ENV_FILE" up -d loongflow

    echo "Waiting for Adapter to be healthy..."
    for i in {1..30}; do
        if curl -s http://localhost:8040/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Adapter is healthy${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ Adapter failed to start${NC}"
            echo "Check logs: docker-compose -f infra/docker-compose-all-adapters.yml logs loongflow"
            exit 1
        fi
        sleep 2
    done
    echo ""

    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Deployment Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "Services:"
    echo "  - LoongFlow Core:    http://localhost:8050"
    echo "  - LoongFlow Adapter: http://localhost:8040"
    echo ""
    echo "Health checks:"
    echo "  - Core:    curl http://localhost:8050/health"
    echo "  - Adapter: curl http://localhost:8040/health"
    echo ""
    echo "Logs:"
    echo "  - Core:    docker-compose -f docker-compose.loongflow-core.yml logs -f"
    echo "  - Adapter: docker-compose -f infra/docker-compose-all-adapters.yml logs -f loongflow"
    echo ""
    echo "Stop services:"
    echo "  - Core:    docker-compose -f docker-compose.loongflow-core.yml down"
    echo "  - Adapter: docker-compose -f infra/docker-compose-all-adapters.yml down"
    echo ""

    # Run validation
    echo -e "${BLUE}Running validation...${NC}"
    echo ""
    sleep 5
    "$SCRIPT_DIR/validate-loongflow-deployment.sh" local
fi

# ============================================================================
# Kubernetes Deployment
# ============================================================================

if [ "$ENVIRONMENT" = "kubernetes" ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Deploying to Kubernetes${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl not found${NC}"
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Kubernetes cluster accessible${NC}"
    echo ""

    # Create secrets
    echo "Creating secrets..."
    kubectl create secret generic loongflow-core-secrets \
        --from-literal=LLM_API_KEY="$LOONGFLOW_LLM_API_KEY" \
        --namespace=loongflow-system \
        --dry-run=client -o yaml | kubectl apply -f -

    kubectl create secret generic loongflow-adapter-secrets \
        --from-literal=LOONGFLOW_API_URL="$LOONGFLOW_API_URL" \
        --namespace=loongflow-system \
        --dry-run=client -o yaml | kubectl apply -f -

    echo -e "${GREEN}✅ Secrets created${NC}"
    echo ""

    # Apply Kubernetes manifests
    echo "Applying Kubernetes manifests..."
    cd "$PROJECT_ROOT"

    # Deploy core service
    kubectl apply -f infra/k8s-loongflow-core.yaml
    echo "Waiting for Core pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=loongflow-core -n loongflow-system --timeout=300s

    echo -e "${GREEN}✅ Core deployed${NC}"
    echo ""

    # Deploy adapter
    kubectl apply -f infra/k8s-loongflow-deployment.yaml
    echo "Waiting for Adapter pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=loongflow-adapter -n loongflow-system --timeout=300s

    echo -e "${GREEN}✅ Adapter deployed${NC}"
    echo ""

    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Deployment Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "Kubernetes resources:"
    echo "  - Namespace: loongflow-system"
    echo "  - Core pods:     kubectl get pods -n loongflow-system -l app=loongflow-core"
    echo "  - Adapter pods:  kubectl get pods -n loongflow-system -l app=loongflow-adapter"
    echo "  - Services:      kubectl get svc -n loongflow-system"
    echo "  - HPA:           kubectl get hpa -n loongflow-system"
    echo ""
    echo "Logs:"
    echo "  - Core:    kubectl logs -f deployment/loongflow-core -n loongflow-system"
    echo "  - Adapter: kubectl logs -f deployment/loongflow-adapter -n loongflow-system"
    echo ""
    echo "Port-forward for testing:"
    echo "  - Adapter: kubectl port-forward svc/loongflow-adapter-service 8040:8040 -n loongflow-system"
    echo ""

    # Note about validation
    echo -e "${YELLOW}⚠️  Note: Run validation with port-forward active${NC}"
    echo ""
    echo "  kubectl port-forward svc/loongflow-adapter-service 8040:8040 -n loongflow-system &"
    echo "  ./scripts/validate-loongflow-deployment.sh local"
    echo ""
fi

echo -e "${GREEN}✅ All done!${NC}"

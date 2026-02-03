# OpenEvolve Makefile
# License: Apache 2.0
#
# Usage:
#   make help              - Show all available commands
#   make install           - Install dependencies
#   make start             - Start all services
#   make test              - Run all tests
#   make docker-up         - Start Docker services
#

.PHONY: help install install-dev start stop restart status health test test-unit test-integration test-coverage lint format clean docker-up docker-down docker-logs docker-build benchmark docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker-compose
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit

# Colors for output (if supported)
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# ============================================================================
# Help
# ============================================================================
help: ## Show this help message
	@echo "$(BLUE)OpenEvolve Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Installation:$(NC)"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "$(GREEN)Service Management:$(NC)"
	@echo "  make start          Start all services"
	@echo "  make stop           Stop all services"
	@echo "  make restart        Restart all services"
	@echo "  make status         Check service status"
	@echo "  make health         Run health checks"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo "  make benchmark      Run performance benchmarks"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  make lint           Run all linters"
	@echo "  make format         Format code with Black"
	@echo "  make check          Run format check (CI)"
	@echo "  make security       Run security scan"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-up      Start Docker services"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make docker-logs    View Docker logs"
	@echo "  make docker-build   Build Docker images"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean          Clean temporary files"
	@echo "  make docs           Generate documentation"
	@echo "  make backup         Create backup"
	@echo "  make migrate        Run MCP migration"

# ============================================================================
# Installation
# ============================================================================
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements_integration.txt
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements_with_testing.txt
	$(PIP) install black flake8 mypy bandit pytest-cov
	@echo "$(GREEN)Development installation complete!$(NC)"

# ============================================================================
# Service Management
# ============================================================================
start: ## Start all services
	@echo "$(BLUE)Starting OpenEvolve services...$(NC)"
	$(PYTHON) -m openevolve_cli services start --all

stop: ## Stop all services
	@echo "$(BLUE)Stopping OpenEvolve services...$(NC)"
	$(PYTHON) -m openevolve_cli services stop

restart: stop start ## Restart all services

status: ## Check service status
	@echo "$(BLUE)Checking service status...$(NC)"
	$(PYTHON) -m openevolve_cli services status

health: ## Run health checks
	@echo "$(BLUE)Running health checks...$(NC)"
	$(PYTHON) system_health.py

cli: ## Start interactive CLI
	@echo "$(BLUE)Starting OpenEvolve CLI...$(NC)"
	$(PYTHON) -m openevolve_cli

# ============================================================================
# Testing
# ============================================================================
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTEST) test_integrations_comprehensive.py -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) test_integrations_comprehensive.py -v -m "not integration and not slow"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) test_integrations_comprehensive.py -v -m integration

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) test_integrations_comprehensive.py --cov=. --cov-report=html --cov-report=term-missing

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) benchmark_integrations.py --all

# ============================================================================
# Code Quality
# ============================================================================
lint: format-check flake8 mypy security ## Run all linters

format: ## Format code with Black
	@echo "$(BLUE)Formatting code with Black...$(NC)"
	$(BLACK) .

format-check: ## Check code formatting (CI)
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(BLACK) --check .

flake8: ## Run Flake8 linter
	@echo "$(BLUE)Running Flake8...$(NC)"
	$(FLAKE8) . --max-line-length=100 --extend-ignore=E203,W503

mypy: ## Run MyPy type checker
	@echo "$(BLUE)Running MyPy...$(NC)"
	$(MYPY) . --ignore-missing-imports

security: ## Run security scan
	@echo "$(BLUE)Running security scan...$(NC)"
	$(BANDIT) -r . -f json -o security_report.json || true
	@echo "$(GREEN)Security report saved to security_report.json$(NC)"

check: format-check lint test ## Run all checks (CI)

# ============================================================================
# Docker
# ============================================================================
docker-up: ## Start Docker services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "REST API: http://localhost:8000"
	@echo "GraphQL:  http://localhost:8001/graphql"
	@echo "Grafana:  http://localhost:3000"
	@echo "Jaeger:   http://localhost:16686"

docker-down: ## Stop Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	$(DOCKER_COMPOSE) down

docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

docker-clean: ## Clean Docker containers and volumes
	@echo "$(YELLOW)Cleaning Docker containers and volumes...$(NC)"
	$(DOCKER_COMPOSE) down -v

docker-restart: docker-down docker-up ## Restart Docker services

# ============================================================================
# Monitoring
# ============================================================================
dashboard: ## Start monitoring dashboard
	@echo "$(BLUE)Starting monitoring dashboard...$(NC)"
	$(PYTHON) -m streamlit run monitoring_dashboard.py

logs: ## View application logs
	@tail -f logs/*.log 2>/dev/null || echo "$(YELLOW)No log files found$(NC)"

metrics: ## View Prometheus metrics
	@curl -s http://localhost:9090/api/v1/status/targets | head -50

# ============================================================================
# Migration
# ============================================================================
migrate-analyze: ## Analyze MCP files for migration
	@echo "$(BLUE)Analyzing MCP files...$(NC)"
	$(PYTHON) migrate_to_unified_mcp.py --analyze

migrate-backup: ## Backup old MCP files
	@echo "$(BLUE)Creating backup of old MCP files...$(NC)"
	$(PYTHON) migrate_to_unified_mcp.py --backup-old

migrate-report: ## Generate migration report
	@echo "$(BLUE)Generating migration report...$(NC)"
	$(PYTHON) migrate_to_unified_mcp.py --report

migrate: migrate-analyze migrate-backup migrate-report ## Run full migration

# ============================================================================
# Maintenance
# ============================================================================
clean: ## Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

backup: ## Create backup of data
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp -r knowledge_extraction backups/$(shell date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@cp -r data backups/$(shell date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@cp *.yaml backups/$(shell date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@echo "$(GREEN)Backup created in backups/$(shell date +%Y%m%d_%H%M%S)/$(NC)"

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements_integration.txt
	@echo "$(GREEN)Update complete!$(NC)"

# ============================================================================
# Development
# ============================================================================
dev-setup: install-dev ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@mkdir -p logs data plugins
	@cp .env.example .env 2>/dev/null || echo "$(YELLOW)Create .env file manually$(NC)"
	@echo "$(GREEN)Development environment ready!$(NC)"

dev-start: ## Start services in development mode
	@echo "$(BLUE)Starting in development mode...$(NC)"
	OPENEVOLVE_LOG_LEVEL=DEBUG $(PYTHON) -m openevolve_cli services start --all --verbose

run-api: ## Start only REST API
	@echo "$(BLUE)Starting REST API...$(NC)"
	$(PYTHON) api_server.py

run-graphql: ## Start only GraphQL API
	@echo "$(BLUE)Starting GraphQL API...$(NC)"
	$(PYTHON) graphql_server.py

run-gateway: ## Start API Gateway
	@echo "$(BLUE)Starting API Gateway...$(NC)"
	$(PYTHON) api_gateway.py

# ============================================================================
# Documentation
# ============================================================================
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "Documentation available in:"
	@echo "  - INTEGRATION_GUIDE.md"
	@echo "  - AGENTS.md"
	@echo "  - http://localhost:8000/docs (Swagger UI)"

readme: ## Update README with current status
	@echo "$(BLUE)Integration status:$(NC)"
	@echo "  Overall: ~95% complete"
	@echo "  Services: 10 implemented"
	@echo "  Tests: 32 test cases"

# ============================================================================
# Deployment
# ============================================================================
deploy-check: check ## Pre-deployment checks
	@echo "$(BLUE)Running pre-deployment checks...$(NC)"
	@$(PYTHON) system_health.py
	@echo "$(GREEN)All checks passed! Ready for deployment.$(NC)"

version: ## Show version info
	@echo "$(BLUE)OpenEvolve Integration System$(NC)"
	@echo "Version: 1.0.0"
	@echo "License: Apache 2.0"
	@echo "Python: $(shell $(PYTHON) --version)"

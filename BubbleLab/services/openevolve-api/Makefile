# OpenEvolve API Makefile
# Development and deployment commands

.PHONY: help dev test lint format clean docker-build docker-up docker-down docker-logs

# Default target
help:
	@echo "OpenEvolve API - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev           - Start development server with hot reload"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"
	@echo "  make docker-logs   - Show Docker logs"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make install       - Install dependencies"

# Development
dev:
	uvicorn openevolve_api.main:app --reload --host 0.0.0.0 --port 8000 --log-level info

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=openevolve_api --cov-report=html --cov-report=term

# Code quality
lint:
	ruff check openevolve_api/
	mypy openevolve_api/

format:
	black openevolve_api/
	ruff check --fix openevolve_api/

# Dependencies
install:
	pip install -r requirements.txt

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Utilities
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf htmlcov/ .coverage

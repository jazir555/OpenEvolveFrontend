.PHONY: help install test deploy clean docker-build docker-up docker-down

help:
	@echo "Sovereign System - Available Commands"
	@echo "======================================"
	@echo "install          - Install dependencies"
	@echo "test             - Run all tests"
	@echo "test-unit        - Run unit tests only"
	@echo "test-integration - Run integration tests"
	@echo "test-benchmarks  - Run performance benchmarks"
	@echo "deploy-dev       - Deploy to development"
	@echo "deploy-staging   - Deploy to staging"
	@echo "deploy-prod      - Deploy to production"
	@echo "docker-build     - Build Docker image"
	@echo "docker-up        - Start Docker containers"
	@echo "docker-down      - Stop Docker containers"
	@echo "clean            - Clean generated files"
	@echo "lint             - Run code quality checks"

install:
	pip install -r requirements.txt

test:
	python -m pytest test_sovereign*.py -v

test-unit:
	python -m pytest test_sovereign_data_models.py test_sovereign_gauntlets.py test_sovereign_quality_assessment.py -v

test-integration:
	python -m pytest test_sovereign_integration.py -v

test-benchmarks:
	python -m pytest test_sovereign_benchmarks.py -v -s

deploy-dev:
	python deploy.py --environment development

deploy-staging:
	python deploy.py --environment staging

deploy-prod:
	python deploy.py --environment production

docker-build:
	docker build -t sovereign-system:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	rm -rf __pycache__ .pytest_cache *.pyc
	rm -f sovereign_system*.db
	rm -f .env logging_config.json
	rm -f start_sovereign.bat start_sovereign.sh

lint:
	flake8 sovereign*.py problem_analyzer.py decomposition_engine.py --max-line-length=100
	black --check sovereign*.py problem_analyzer.py decomposition_engine.py

format:
	black sovereign*.py problem_analyzer.py decomposition_engine.py

health-check:
	python -c "from sovereign_reliability import get_health_monitor; m = get_health_monitor(); print(m.run_health_checks())"

backup:
	cp sovereign_system.db sovereign_system_backup_$$(date +%Y%m%d_%H%M%S).db

restore:
	@echo "Usage: make restore BACKUP=sovereign_system_backup_YYYYMMDD_HHMMSS.db"
	@if [ -n "$(BACKUP)" ]; then cp $(BACKUP) sovereign_system.db; echo "Restored from $(BACKUP)"; fi

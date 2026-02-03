.PHONY: help install test test-watch test-coverage test-contract validate clean lint

# Default target
help:
	@echo "Z3 Adapter Contract Tests - Available Commands:"
	@echo ""
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-watch    - Run tests in watch mode"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make test-contract - Run contract tests only"
	@echo "  make validate      - Run contract validation (alias for test-contract)"
	@echo "  make clean         - Remove build artifacts and coverage"
	@echo "  make lint          - Run TypeScript type checking"
	@echo ""

install:
	npm install

test:
	npm test

test-watch:
	npm run test:watch

test-coverage:
	npm run test:coverage
	@echo ""
	@echo "Coverage report generated in ./coverage/"

test-contract:
	npm run test:contract

validate: test-contract

clean:
	rm -rf dist/
	rm -rf coverage/
	rm -rf *.tsbuildinfo
	@echo "Clean complete"

lint:
	npx tsc --noEmit
	@echo "TypeScript type checking complete"

#!/bin/bash
# Validation script for bubblelab-converted fixes
# Follows Federation Constitution Section 4: The Proof of Work (Phase 1: The Probe)

set -e

echo "=== Validating BubbleLab-Converted Fixes ==="
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
        return 0
    else
        echo -e "${RED}✗${NC} $1 NOT FOUND"
        return 1
    fi
}

# Function to check file contains string
check_content() {
    if grep -q "$2" "$1"; then
        echo -e "${GREEN}✓${NC} $1 contains: $2"
        return 0
    else
        echo -e "${RED}✗${NC} $1 missing: $2"
        return 1
    fi
}

echo "1. Checking created files..."
check_file "src/lib/openevolveApi.test.ts"
check_file "vitest.config.ts"
echo

echo "2. Checking modified files..."
check_file "src/lib/openevolveApi.ts"
check_file "src/components/openevolve/main/GithubIntegrationTab.tsx"
check_file "src/components/openevolve/main/OpenEvolveApp.tsx"
check_file "package.json"
echo

echo "3. Checking imports in openevolveApi.ts..."
check_content "src/lib/openevolveApi.ts" "import.*retryWithBackoff.*from.*retry"
check_content "src/lib/openevolveApi.ts" "import.*CircuitBreaker.*from.*circuit-breaker"
check_content "src/lib/openevolveApi.ts" "import.*apiLogger.*from.*structuredLogger"
echo

echo "4. Checking circuit breaker configuration..."
check_content "src/lib/openevolveApi.ts" "openevolveCircuitBreaker"
check_content "src/lib/openevolveApi.ts" "threshold: 5"
check_content "src/lib/openevolveApi.ts" "timeout_ms: 60000"
echo

echo "5. Checking retry logic..."
check_content "src/lib/openevolveApi.ts" "retryWithBackoff"
check_content "src/lib/openevolveApi.ts" "max_retries:"
echo

echo "6. Checking GitHub API timeout..."
check_content "src/components/openevolve/main/GithubIntegrationTab.tsx" "AbortController"
check_content "src/components/openevolve/main/GithubIntegrationTab.tsx" "controller.abort"
check_content "src/components/openevolve/main/GithubIntegrationTab.tsx" "30000"
echo

echo "7. Checking environment variable configuration..."
check_content "src/components/openevolve/main/GithubIntegrationTab.tsx" "process.env?.GITHUB_API_BASE"
check_content "src/lib/openevolveApi.ts" "process.env?.DEFAULT_REQUEST_TIMEOUT"
check_content "src/lib/openevolveApi.ts" "process.env?.MAX_RETRIES"
echo

echo "8. Checking structured logging..."
check_content "src/components/openevolve/main/OpenEvolveApp.tsx" "import.*apiLogger"
check_content "src/components/openevolve/main/OpenEvolveApp.tsx" "apiLogger.error"
check_content "src/components/openevolve/main/GithubIntegrationTab.tsx" "apiLogger.warn"
echo

echo "9. Checking test configuration..."
check_content "package.json" "\"test\":"
check_content "package.json" "\"test:contract\":"
check_content "vitest.config.ts" "export default defineConfig"
echo

echo "10. Checking contract tests..."
check_content "src/lib/openevolveApi.test.ts" "describe.*OpenEvolve API Contract Tests"
check_content "src/lib/openevolveApi.test.ts" "describe.*Teams API"
check_content "src/lib/openevolveApi.test.ts" "describe.*Evolution API"
check_content "src/lib/openevolveApi.test.ts" "describe.*Adversarial Testing API"
echo

echo "=== Validation Complete ==="
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run 'npm install' to install test dependencies"
echo "2. Run 'npm run test:contract' to execute contract tests"
echo "3. Set environment variables in .env file"
echo "4. Run 'npm run build' to verify TypeScript compilation"

#!/bin/bash
# P3 Final Wave - Quick Start Script
# This script sets up the testing infrastructure and begins implementation

set -e  # Exit on error

echo "=========================================="
echo "P3 Final Wave - Testing & Production Prep"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="C:/Users/mmeadow/Documents/OpenEvolve/Frontend/BubbleLab"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Step 1: Installing coverage dependencies...${NC}"
pnpm add -D -w @vitest/coverage-v8
echo -e "${GREEN}✓ Coverage dependencies installed${NC}"
echo ""

echo -e "${YELLOW}Step 2: Creating vitest configuration files...${NC}"

# Create vitest config for bubble-runtime
cat > packages/bubble-runtime/vitest.config.ts << 'EOF'
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'dist/',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/types/',
        '**/fixtures/',
        '**/dist/',
      ],
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
EOF

# Create vitest config for bubble-core
cat > packages/bubble-core/vitest.config.ts << 'EOF'
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    setupFiles: ['./src/tests/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'dist/',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/types/',
        '**/fixtures/',
        '**/dist/',
      ],
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
EOF

echo -e "${GREEN}✓ Vitest configuration files created${NC}"
echo ""

echo -e "${YELLOW}Step 3: Running baseline coverage report...${NC}"
echo "This may take a few minutes..."
echo ""

# Run coverage for bubble-runtime
echo "Testing bubble-runtime..."
cd packages/bubble-runtime
pnpm test:coverage || true
cd ../..

# Run coverage for bubble-core
echo "Testing bubble-core..."
cd packages/bubble-core
pnpm test:coverage || true
cd ../..

echo -e "${GREEN}✓ Baseline coverage reports generated${NC}"
echo ""
echo "Check the coverage reports:"
echo "  - packages/bubble-runtime/coverage/index.html"
echo "  - packages/bubble-core/coverage/index.html"
echo ""

echo -e "${YELLOW}Step 4: Creating test file structure...${NC}"

# Create directories for new tests
mkdir -p packages/bubble-runtime/src/utils
mkdir -p packages/bubble-core/src/bubbles/service-bubble/tests
mkdir -p packages/bubble-core/src/bubbles/tool-bubble/tests

echo -e "${GREEN}✓ Test file structure created${NC}"
echo ""

echo -e "${YELLOW}Step 5: Creating placeholder test files...${NC}"

# Create placeholder test files with templates
cat > packages/bubble-runtime/src/utils/validators.test.ts << 'EOF'
import { describe, it, expect } from 'vitest';

describe('Validators', () => {
  describe('validateEmail', () => {
    it('should accept valid email addresses', () => {
      // TODO: Implement email validation
      expect(true).toBe(true);
    });
  });

  describe('validateUrl', () => {
    it('should accept valid URLs', () => {
      // TODO: Implement URL validation
      expect(true).toBe(true);
    });
  });

  describe('validateTimestamp', () => {
    it('should accept valid ISO-8601 timestamps', () => {
      // TODO: Implement timestamp validation
      expect(true).toBe(true);
    });
  });
});
EOF

cat > packages/bubble-runtime/src/utils/error-handlers.test.ts << 'EOF'
import { describe, it, expect } from 'vitest';

describe('Error Handlers', () => {
  describe('categorizeError', () => {
    it('should categorize network errors as transient', () => {
      // TODO: Implement error categorization
      expect(true).toBe(true);
    });
  });

  describe('isRetryableError', () => {
    it('should identify retryable errors', () => {
      // TODO: Implement retryable error detection
      expect(true).toBe(true);
    });
  });
});
EOF

cat > packages/bubble-runtime/src/utils/retry-logic.test.ts << 'EOF'
import { describe, it, expect } from 'vitest';

describe('Retry Logic', () => {
  describe('calculateBackoff', () => {
    it('should calculate exponential backoff', () => {
      // TODO: Implement backoff calculation
      expect(true).toBe(true);
    });
  });

  describe('CircuitBreaker', () => {
    it('should start in closed state', () => {
      // TODO: Implement circuit breaker
      expect(true).toBe(true);
    });
  });
});
EOF

cat > packages/bubble-runtime/src/utils/cache.test.ts << 'EOF'
import { describe, it, expect } from 'vitest';

describe('Cache', () => {
  describe('Basic Operations', () => {
    it('should set and get values', () => {
      // TODO: Implement cache tests
      expect(true).toBe(true);
    });
  });
});
EOF

cat > packages/bubble-runtime/src/utils/connection-pool.test.ts << 'EOF'
import { describe, it, expect } from 'vitest';

describe('ConnectionPool', () => {
  describe('Connection Acquisition', () => {
    it('should create new connection when pool is empty', () => {
      // TODO: Implement connection pool tests
      expect(true).toBe(true);
    });
  });
});
EOF

echo -e "${GREEN}✓ Placeholder test files created${NC}"
echo ""

echo -e "${YELLOW}Step 6: Creating documentation directories...${NC}"

# Create documentation structure
mkdir -p docs/RUNBOOKS

# Create runbook templates
cat > docs/RUNBOOKS/DEPLOYMENT_RUNBOOK.md << 'EOF'
# Deployment Runbook

## Prerequisites
- [ ] All tests passing
- [ ] Coverage targets met (80%+)
- [ ] Security scan clean
- [ ] Environment variables configured

## Deployment Steps

### 1. Prepare Release
```bash
# Version bump
npm version [patch|minor|major]

# Build packages
pnpm build

# Run tests
pnpm test
```

### 2. Deploy to Staging
```bash
# Deploy to staging environment
pnpm deploy:staging

# Run smoke tests
pnpm test:smoke:staging
```

### 3. Deploy to Production
```bash
# Create production build
pnpm build:prod

# Deploy with zero-downtime
pnpm deploy:prod --zero-downtime

# Verify deployment
pnpm verify:prod
```

## Rollback Procedure
```bash
# Identify bad version
git log --oneline

# Rollback to previous version
git revert HEAD

# Hotfix rollback
pnpm deploy:rollback --version <version>
```
EOF

cat > docs/RUNBOOKS/INCIDENT_RESPONSE.md << 'EOF'
# Incident Response Runbook

## Severity Levels
- **P0 - Critical**: System down, total outage
- **P1 - High**: Major functionality broken
- **P2 - Medium**: Partial degradation
- **P3 - Low**: Minor issues

## P0 Incident Response

### Immediate Actions (5 mins)
1. Acknowledge alert
2. Join incident channel
3. Identify scope of impact

### Investigation (15 mins)
1. Check dashboards
2. Review logs
3. Identify root cause

### Mitigation (30 mins)
1. Implement temporary fix
2. Rollback if necessary
3. Communicate status

## Common Incidents

### Database Connection Pool Exhausted
**Symptoms:** Timeout errors, slow queries
**Mitigation:**
  1. Increase pool size
  2. Restart affected services
  3. Kill long-running queries

### API Rate Limit Exceeded
**Symptoms:** 429 errors, rejected requests
**Mitigation:**
  1. Implement backpressure
  2. Enable caching
  3. Scale horizontally
EOF

echo -e "${GREEN}✓ Documentation structure created${NC}"
echo ""

echo "=========================================="
echo -e "${GREEN}P3 Final Wave Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Review baseline coverage reports:"
echo "   - open packages/bubble-runtime/coverage/index.html"
echo "   - open packages/bubble-core/coverage/index.html"
echo ""
echo "2. Start implementing tests:"
echo "   - Edit packages/bubble-runtime/src/utils/validators.test.ts"
echo "   - Edit packages/bubble-runtime/src/utils/error-handlers.test.ts"
echo "   - Edit packages/bubble-runtime/src/utils/retry-logic.test.ts"
echo "   - Edit packages/bubble-runtime/src/utils/cache.test.ts"
echo "   - Edit packages/bubble-runtime/src/utils/connection-pool.test.ts"
echo ""
echo "3. Run tests while developing:"
echo "   cd packages/bubble-runtime"
echo "   pnpm test --watch"
echo ""
echo "4. Check coverage as you go:"
echo "   pnpm test:coverage"
echo ""
echo "5. Refer to implementation guide:"
echo "   - P3_IMPLEMENTATION_GUIDE.md"
echo "   - P3_FINAL_WAVE_STATUS.md"
echo ""
echo -e "${YELLOW}Estimated time to complete Phase 1 (Infrastructure + Common Utilities): 7-8 hours${NC}"
echo ""

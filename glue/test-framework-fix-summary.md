# Test Framework Dependency Fix Summary

## Problem
Multiple test files were importing from `vitest` which was not installed in the project, causing import errors.

## Solution
Replaced all `vitest` imports with `@jest/globals` imports across the codebase.

## Files Fixed

### 1. RAGBits Adapter
- **File:** `glue/adapters/ragbits/bubblelabs-ragbits-plugin/src/tests/contract/ragbits-api.test.ts`
- **Change:** `import { describe, it, expect, beforeAll } from 'vitest'` → `from '@jest/globals'`

### 2. Datapizza Adapter
- **File:** `glue/adapters/datapizza/datapizza-bubblelab-plugin/src/tests/contract/datapizza-api.test.ts`
- **Change:** `import { describe, it, expect, beforeAll } from 'vitest'` → `from '@jest/globals'`

### 3. Bubblelab Adapter - Library Tests
- **File:** `glue/adapters/bubblelab/src/lib/openevolveApi.test.ts`
- **Change:** `import { describe, it, expect, beforeAll, afterEach } from 'vitest'` → `from '@jest/globals'`

### 4. Bubblelab Adapter - API Contract Tests
- **File:** `glue/adapters/bubblelab/src/tests/api-contracts/gauntlet-decomposition-api.test.ts`
- **Change:** `import { describe, it, expect } from 'vitest'` → `from '@jest/globals'`

### 5. Bubblelab Adapter - OpenEvolve API Contract Tests
- **File:** `glue/adapters/bubblelab/src/tests/contract/openevolve-api.test.ts`
- **Change:** `import { describe, it, expect } from 'vitest'` → `from '@jest/globals'`

### 6. Bubblelab Adapter - Workflow Orchestrator Tests
- **File:** `glue/adapters/bubblelab/src/tests/contract/workflow-orchestrator.test.ts`
- **Change:** `import { describe, it, expect, beforeEach, afterEach } from 'vitest'` → `from '@jest/globals'`

### 7. Bubblelab Adapter - E2E Integration Tests
- **File:** `glue/adapters/bubblelab/src/tests/integration/e2e-integration.test.ts`
- **Change:** `import { describe, it, expect, beforeEach, afterEach, beforeAll } from 'vitest'` → `from '@jest/globals'`

## Verification
All test files in the glue directory now use `@jest/globals` instead of `vitest`. No vitest imports remain in project source files (excluding node_modules).

## Import Mapping Reference
- `vitest` → `@jest/globals`
- Test functions remain the same: `describe`, `it`, `expect`, `beforeAll`, `afterAll`, `beforeEach`, `afterEach`

## Date
2026-02-22

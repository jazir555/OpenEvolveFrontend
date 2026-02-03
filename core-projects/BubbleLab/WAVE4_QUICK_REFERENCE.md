# Wave 4 Security Fixes - Quick Reference Guide

**Date:** 2026-01-17
**Purpose:** Quick reference for applying security fixes to remaining 41 workflow files

---

## 🚨 Status: Only 3/44 Files Fixed

**Verified Fixed:**
- ✅ container-health-monitor.ts
- ✅ database-backup-validator.ts
- ✅ log-aggregation-analyzer.ts

**Still Need Fixing (41 files):**
- ❌ All other templates and examples

---

## 📋 Security Fix Checklist

For each workflow file, complete these 5 steps:

### Step 1: Import Security Utilities

Add at top of file (after existing imports):

```typescript
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  buildParameterizedQuery,
} from '../security-utils';
```

### Step 2: Add Environment Variable Validation

Add after imports, before class definition:

```typescript
// Define your schemas
const ApiKeySchema = z.string().min(32).max(256);

// Validate at startup
validateEnvironment({
  required: ['API_KEY', 'OTHER_REQUIRED_VARS'],
  schemas: {
    API_KEY: ApiKeySchema,
  },
});
```

### Step 3: Add Input Validation Schemas

Add after imports:

```typescript
// Example for container IDs
const ContainerIdSchema = z.string()
  .regex(/^[a-f0-9]{12,}$/, 'Invalid container ID format');

// Example for service names
const ServiceNameSchema = z.string()
  .min(1)
  .max(255)
  .regex(/^[a-zA-Z0-9_-]+$/, 'Invalid service name');
```

### Step 4: Add Security to handle() Method

```typescript
async handle(payload: WebhookEvent): Promise<Result> {
  // Generate correlation ID
  const correlationId = generateCorrelationId();

  // Create logger
  const logger = new StructuredLogger('workflow-name');

  // Check rate limit
  if (!this.rateLimiter.checkLimit(correlationId)) {
    throw new Error('Rate limit exceeded. Please try again later.');
  }

  // Authenticate
  const authContext = authenticateRequest(
    payload.headers?.['x-api-key'],
    process.env.API_KEY,
    { correlationId, ip: payload.headers?.['x-forwarded-for'] }
  );
  requireAuthentication(authContext);

  // ... rest of your code
}
```

### Step 5: Add Rate Limiter to Class

```typescript
export class MyWorkflow extends BubbleFlow<'webhook/http'> {
  private rateLimiter = new RateLimiter({
    maxRequests: 100,  // Adjust based on your workflow
    windowMs: 60000,   // 1 minute
  });

  // ... rest of class
}
```

---

## 🔍 Common Patterns

### Pattern 1: SQL Queries

❌ **BEFORE (Vulnerable):**
```typescript
const query = `SELECT * FROM logs WHERE timestamp > '${userInput}'`;
```

✅ **AFTER (Secure):**
```typescript
const query = buildParameterizedQuery(
  `SELECT * FROM logs WHERE timestamp > $1`,
  [userInput]
);
```

### Pattern 2: Container IDs

❌ **BEFORE (Vulnerable):**
```typescript
url: `${process.env.DOCKER_HOST}/containers/${containerId}/json`
```

✅ **AFTER (Secure):**
```typescript
const sanitizedId = InputValidator.validateContainerId(containerId);
url: `${process.env.DOCKER_HOST}/containers/${sanitizedId}/json`
```

### Pattern 3: Error Handling

❌ **BEFORE (Vulnerable):**
```typescript
} catch (error) {
  console.error('Failed:', error);
  throw error; // Leaks stack traces
}
```

✅ **AFTER (Secure):**
```typescript
} catch (error) {
  logger.error({
    msg: 'Operation failed',
    correlationId,
  }, error);
  throw new Error('Operation failed'); // Generic message
}
```

### Pattern 4: Webhook Payloads

❌ **BEFORE (Vulnerable):**
```typescript
async handle(payload: WebhookEvent & MyData): Promise<Result> {
  const { field1, field2 } = payload; // No validation
}
```

✅ **AFTER (Secure):**
```typescript
const MyDataSchema = z.object({
  field1: z.string().min(1).max(255),
  field2: z.number().int().min(0),
});

async handle(payload: WebhookEvent): Promise<Result> {
  const validated = MyDataSchema.parse(payload);
  const { field1, field2 } = validated;
}
```

---

## 📁 Template Files to Copy From

### Best Overall Template:
**log-aggregation-analyzer.ts**
- Perfect use of security-utils
- Clean, readable code
- All security patterns demonstrated
- Import this file and copy the patterns

### For Cron/Scheduled Workflows:
**container-health-monitor.ts**
- Good pattern for scheduled tasks
- Rate limiting with correlation IDs
- Container validation patterns

### For Database Operations:
**database-backup-validator.ts**
- Parameterized queries
- Database name validation
- Multi-step transaction patterns

---

## 🎯 Rate Limiting Guidelines

Choose appropriate limits based on workflow type:

| Workflow Type | Max Requests | Window | Notes |
|--------------|--------------|--------|-------|
| Cron/Scheduled | 10-60 | 1 hour | Low frequency |
| Webhook (Low Volume) | 100 | 1 minute | Normal API |
| Webhook (High Volume) | 1000 | 1 minute | High-traffic |
| Admin Operations | 10 | 1 minute | Sensitive ops |
| Backup/Maintenance | 10 | 1 hour | Resource-intensive |

---

## ⚠️ Common Mistakes to Avoid

### Mistake 1: Forgetting to Import security-utils
```typescript
// ❌ WRONG - Will crash
validateEnvironment({ required: ['API_KEY'] });

// ✅ CORRECT
import { validateEnvironment } from '../security-utils';
validateEnvironment({ required: ['API_KEY'] });
```

### Mistake 2: Validating After Use
```typescript
// ❌ WRONG - Already used unsafe input
const url = `${process.env.DOCKER_HOST}/containers/${containerId}/json`;
InputValidator.validateContainerId(containerId);

// ✅ CORRECT - Validate first
const sanitizedId = InputValidator.validateContainerId(containerId);
const url = `${process.env.DOCKER_HOST}/containers/${sanitizedId}/json`;
```

### Mistake 3: Hardcoding Rate Limits
```typescript
// ❌ WRONG - Not configurable
private readonly MAX_REQUESTS = 100;

// ✅ CORRECT - Environment-based
private readonly MAX_REQUESTS = parseInt(process.env.MAX_REQUESTS || '100', 10);
```

### Mistake 4: Exposing Errors
```typescript
// ❌ WRONG - Leaks sensitive data
} catch (error) {
  throw new Error(`Database failed: ${error.message}`);
}

// ✅ CORRECT - Generic message
} catch (error) {
  logger.error({ msg: 'Database operation failed' }, error);
  throw new Error('Database operation failed');
}
```

---

## ✅ Verification Checklist

After fixing each file, verify:

- [ ] Imports security-utils functions
- [ ] Validates environment variables at startup
- [ ] Has Zod schemas for all inputs
- [ ] Authenticates requests with API key
- [ ] Rate limits requests
- [ ] Sanitizes all error messages
- [ ] Uses parameterized SQL queries
- [ ] Validates container IDs/commands
- [ ] Logs with correlation IDs
- [ ] No hardcoded credentials

---

## 🚀 Quick Start: Fix a File in 10 Minutes

### Minute 1-2: Add Imports
```typescript
import { z } from 'zod';
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
} from '../security-utils';
```

### Minute 3-4: Add Validation
```typescript
const ApiKeySchema = z.string().min(32).max(256);

validateEnvironment({
  required: ['API_KEY'],
  schemas: { API_KEY: ApiKeySchema },
});
```

### Minute 5-6: Add Rate Limiter
```typescript
export class MyWorkflow extends BubbleFlow<'webhook/http'> {
  private rateLimiter = new RateLimiter({
    maxRequests: 100,
    windowMs: 60000,
  });
```

### Minute 7-8: Update handle() Method
```typescript
async handle(payload: WebhookEvent): Promise<Result> {
  const correlationId = generateCorrelationId();
  const logger = new StructuredLogger('my-workflow');

  if (!this.rateLimiter.checkLimit(correlationId)) {
    throw new Error('Rate limit exceeded');
  }

  const authContext = authenticateRequest(
    payload.headers?.['x-api-key'],
    process.env.API_KEY,
    { correlationId }
  );
  requireAuthentication(authContext);

  // ... existing code
}
```

### Minute 9-10: Test
- Test with missing API key (should get 401)
- Test with invalid API key (should get 401)
- Test rate limiting (should get 429 after limit)
- Test with valid API key (should work)

---

## 📊 Progress Tracker

### Files Fixed (3/44 = 6.8%)

**Infrastructure:**
- [✅] container-health-monitor.ts
- [✅] database-backup-validator.ts
- [✅] log-aggregation-analyzer.ts
- [ ] service-deployment-automation.ts
- [ ] resource-scaling-automation.ts
- [ ] service-dependency-scanner.ts
- [ ] distributed-tracing-analyzer.ts

**Development:**
- [ ] code-review-automation.ts
- [ ] test-execution-reporter.ts
- [ ] dependency-update-automation.ts
- [ ] documentation-generator.ts
- [ ] deployment-pipeline-orchestrator.ts
- [ ] automated-changelog-generator.ts
- [ ] security-vulnerability-scanner.ts

**LLM Operations:**
- [ ] prompt-testing-validator.ts
- [ ] model-performance-benchmark.ts
- [ ] token-usage-monitor.ts
- [ ] ai-response-quality-assessor.ts
- [ ] multi-model-comparison-tester.ts
- [ ] prompt-optimizer.ts

**Examples:** (24 files)
- [ ] All example workflows need fixing

---

## 🔧 Troubleshooting

### Issue: "Cannot find module '../security-utils'"

**Solution:** Make sure security-utils.ts is in the parent directory of your workflow file.

### Issue: "validateEnvironment is not a function"

**Solution:** Check you're importing it correctly:
```typescript
import { validateEnvironment } from '../security-utils'; // ✅
import validateEnvironment from '../security-utils'; // ❌
```

### Issue: "Rate limit exceeded" during testing

**Solution:** Use different correlation IDs for each test, or increase the limit temporarily.

### Issue: "Unauthorized: Invalid API key" even with correct key

**Solution:**
1. Check API_KEY environment variable is set
2. Verify key is 32+ characters long
3. Check you're sending it in `x-api-key` header

---

## 📚 Additional Resources

- **Full Verification Report:** `WAVE4_SECURITY_VERIFICATION.md`
- **Executive Summary:** `WAVE4_EXECUTIVE_SUMMARY.md`
- **Security Utilities:** `templates/security-utils.ts`
- **Model Implementation:** `templates/infrastructure/log-aggregation-analyzer.ts`

---

## 🎯 Target: 100% Complete by [DATE]

**Current:** 3/44 files (6.8%)
**Target:** 44/44 files (100%)
**Remaining:** 41 files

**At 10 min/file:** 410 minutes = ~7 hours
**At 20 min/file:** 820 minutes = ~14 hours
**At 30 min/file:** 1230 minutes = ~20 hours

**Recommended:** Allocate 20-25 hours for thorough fixes including testing.

---

**Last Updated:** 2026-01-17
**Next Review:** After all 41 files are fixed
**Questions:** Refer to full verification report or ask security team

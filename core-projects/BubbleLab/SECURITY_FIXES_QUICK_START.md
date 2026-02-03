# Security Fixes Quick Start Guide

**Purpose:** Quickly apply Wave 2 security fixes to remaining BubbleLab workflow files

---

## Pattern: Apply Security Fixes in 5 Minutes

### Step 1: Add Imports (30 seconds)

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
  buildParameterizedQuery,
} from '../security-utils';
```

### Step 2: Add Startup Validation (1 minute)

Add this at the module level (before the class definition):

```typescript
// Environment variable validation
validateEnvironment({
  required: ['API_KEY', 'YOUR_SPECIFIC_VAR'],
  schemas: {
    API_KEY: z.string().min(32).max(256),
  },
});
```

### Step 3: Update Interface (30 seconds)

Add `correlationId` to your result interface:

```typescript
interface YourResult {
  // ... existing fields
  correlationId: string; // ADD THIS
}
```

### Step 4: Add Class Properties (1 minute)

Add to your workflow class:

```typescript
export class YourWorkflow extends BubbleFlow<'webhook/http'> {
  // Add these properties
  private logger = new StructuredLogger('your-workflow-name');
  private rateLimiter = new RateLimiter({
    maxRequests: 100,
    windowMs: 60000,
  });

  // ... rest of class
}
```

### Step 5: Update handle() Method (2 minutes)

Replace the start of your `handle()` method:

```typescript
async handle(payload: WebhookEvent): Promise<YourResult> {
  // ADD THESE LINES AT THE START
  const correlationId = generateCorrelationId();
  this.logger = this.logger.child({ correlationId });

  // Rate limiting
  if (!this.rateLimiter.checkLimit(correlationId)) {
    throw new Error('Rate limit exceeded. Please try again later.');
  }

  // Authentication
  const authContext = authenticateRequest(
    payload.headers?.['x-api-key'],
    process.env.API_KEY,
    { correlationId, ip: payload.headers?.['x-forwarded-for'] }
  );
  requireAuthentication(authContext);

  // ... rest of your existing code
```

### Step 6: Update Return Statement (30 seconds)

Add correlationId to your return value:

```typescript
  return {
    // ... existing fields
    correlationId, // ADD THIS
  };
}
```

### Step 7: Add Input Validation (if webhook) (1 minute)

Add validation schema for webhook payload:

```typescript
// Add after imports
const YourPayloadSchema = z.object({
  field1: z.string().min(1).max(255),
  field2: z.number().int().min(0),
});

// Add at start of handle(), after authentication
const validated = YourPayloadSchema.parse(payload);
```

### Step 8: Fix SQL Queries (if any) (1 minute)

Find and replace SQL queries:

**BEFORE:**
```typescript
const query = `SELECT * FROM table WHERE field = '${userInput}'`;
```

**AFTER:**
```typescript
const query = buildParameterizedQuery(
  'SELECT * FROM table WHERE field = $1',
  [userInput]
);
```

Then update PostgreSQLBubble call:

**BEFORE:**
```typescript
new PostgreSQLBubble({
  connectionString: process.env.DATABASE_URL,
  query: `SELECT * FROM table WHERE field = '${userInput}'`,
})
```

**AFTER:**
```typescript
new PostgreSQLBubble({
  connectionString: process.env.DATABASE_URL,
  query: query.query,
  params: query.params,
})
```

### Step 9: Sanitize Error Messages (1 minute)

Replace generic error logging:

**BEFORE:**
```typescript
} catch (error) {
  console.error('Operation failed:', error);
}
```

**AFTER:**
```typescript
} catch (error) {
  this.logger.error({
    msg: 'Operation failed',
    correlationId,
  }, error);
  // Don't re-throw for non-critical operations
}
```

---

## Complete Example: Before and After

### BEFORE (Vulnerable)

```typescript
import { BubbleFlow, PostgreSQLBubble } from '@bubblelab/bubble-core';

interface Result {
  success: boolean;
  data: any[];
}

export class MyWorkflow extends BubbleFlow<'webhook/http'> {
  readonly name = 'My Workflow';

  async handle(payload: any): Promise<Result> {
    const startTime = Date.now();

    // No authentication!
    // No rate limiting!
    // No input validation!

    const query = `SELECT * FROM data WHERE id = '${payload.id}'`; // SQL INJECTION!

    const result = await new PostgreSQLBubble({
      connectionString: process.env.DATABASE_URL,
      query, // Vulnerable!
    }).action();

    console.log('Query result:', result); // Logs sensitive data!

    return {
      success: true,
      data: result.data.rows,
    };
  }
}
```

### AFTER (Secure)

```typescript
import { BubbleFlow, PostgreSQLBubble, type WebhookEvent } from '@bubblelab/bubble-core';
import { z } from 'zod';
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  StructuredLogger,
  generateCorrelationId,
  buildParameterizedQuery,
} from '../security-utils';

// Validate environment at startup
validateEnvironment({
  required: ['API_KEY', 'DATABASE_URL'],
  schemas: {
    API_KEY: z.string().min(32).max(256),
  },
});

// Input validation schema
const MyPayloadSchema = z.object({
  id: z.string().regex(/^[0-9]+$/, 'ID must be numeric'),
});

interface Result {
  success: boolean;
  data: any[];
  correlationId: string;
}

export class MyWorkflow extends BubbleFlow<'webhook/http'> {
  readonly name = 'My Workflow';

  // Security utilities
  private logger = new StructuredLogger('my-workflow');
  private rateLimiter = new RateLimiter({
    maxRequests: 100,
    windowMs: 60000,
  });

  async handle(payload: WebhookEvent): Promise<Result> {
    // Generate correlation ID
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    // Rate limiting
    if (!this.rateLimiter.checkLimit(correlationId)) {
      throw new Error('Rate limit exceeded');
    }

    // Authentication
    const auth = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId }
    );
    requireAuthentication(auth);

    // Input validation
    const validated = MyPayloadSchema.parse(payload);

    // SQL injection prevention
    const query = buildParameterizedQuery(
      'SELECT * FROM data WHERE id = $1',
      [validated.id]
    );

    // Execute with error handling
    let result;
    try {
      result = await new PostgreSQLBubble({
        connectionString: process.env.DATABASE_URL,
        query: query.query,
        params: query.params,
      }).action();
    } catch (error) {
      this.logger.error({
        msg: 'Database query failed',
      }, error);
      throw new Error('Query execution failed');
    }

    // Return with correlation ID
    return {
      success: true,
      data: result.data.rows,
      correlationId,
    };
  }
}
```

---

## Common Patterns by File Type

### Infrastructure Templates

**Required Environment Variables:**
- `API_KEY`
- `DOCKER_HOST` or `KUBERNETES_API`
- `SLACK_WEBHOOK_URL` (optional)

**Input Validation Schemas:**
```typescript
const ContainerIdSchema = z.string().regex(/^[a-f0-9]{12,}$/);
const ServiceNameSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/);
const NamespaceSchema = z.string().min(1).max(253).regex(/^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/);
```

**Rate Limits:** 100 req/min (infrastructure operations)

---

### Development Templates

**Required Environment Variables:**
- `API_KEY`
- `GITHUB_TOKEN` or `GITLAB_TOKEN`
- `SLACK_WEBHOOK_URL` (optional)

**Input Validation Schemas:**
```typescript
const RepositorySchema = z.string().regex(/^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$/);
const BranchSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/);
const PullRequestSchema = z.number().int().positive();
```

**Rate Limits:** 60 req/min (CI/CD operations)

---

### LLM Operations Templates

**Required Environment Variables:**
- `API_KEY`
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- Database URL (for storing results)

**Input Validation Schemas:**
```typescript
const PromptSchema = z.string().min(1).max(10000);
const ModelNameSchema = z.string().min(1).max(100);
const TokenCountSchema = z.number().int().min(0).max(100000);
```

**Rate Limits:** 10 req/min (expensive LLM operations)

---

## Testing Your Fixes

### 1. Unit Test Template

```typescript
describe('MyWorkflow Security', () => {
  it('should reject requests without API key', async () => {
    const workflow = new MyWorkflow();

    await expect(
      workflow.handle({ headers: {} } as any)
    ).rejects.toThrow('Unauthorized');
  });

  it('should reject invalid input', async () => {
    const workflow = new MyWorkflow();

    await expect(
      workflow.handle({
        headers: { 'x-api-key': process.env.API_KEY },
        id: 'malicious; DROP TABLE users--',
      } as any)
    ).rejects.toThrow();
  });

  it('should enforce rate limits', async () => {
    const workflow = new MyWorkflow();

    // Send 101 requests
    const promises = Array(101).fill(null).map(() =>
      workflow.handle({
        headers: { 'x-api-key': process.env.API_KEY },
        id: '123',
      } as any)
    );

    const results = await Promise.allSettled(promises);

    // Last request should fail
    expect(results[100].status).toBe('rejected');
  });
});
```

### 2. Integration Test Template

```typescript
describe('MyWorkflow Integration', () => {
  it('should handle valid request end-to-end', async () => {
    const response = await fetch('http://localhost:3000/api/my-workflow', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.API_KEY!,
      },
      body: JSON.stringify({ id: '123' }),
    });

    expect(response.status).toBe(200);
    const data = await response.json();
    expect(data).toHaveProperty('correlationId');
  });
});
```

---

## Checklist for Each File

Use this checklist for each file you fix:

- [ ] Import security utilities
- [ ] Add environment variable validation
- [ ] Add API_KEY to required vars
- [ ] Add rate limiter property
- [ ] Add logger property
- [ ] Add correlation ID generation
- [ ] Add rate limit check
- [ ] Add authentication check
- [ ] Add input validation schema (if webhook)
- [ ] Fix SQL queries to use parameterization (if any)
- [ ] Sanitize error messages
- [ ] Add correlation ID to result interface
- [ ] Add correlation ID to return value
- [ ] Replace console.log with logger.info
- [ ] Replace console.error with logger.error

---

## Time Estimates

| File Type | Base Fixes | Input Validation | SQL Fixes | Total |
|-----------|------------|------------------|-----------|-------|
| Simple workflow | 3 min | 1 min | 0 min | **4 min** |
| Webhook workflow | 3 min | 2 min | 0 min | **5 min** |
| Database workflow | 3 min | 1 min | 2 min | **6 min** |
| Complex workflow | 3 min | 2 min | 2 min | **7 min** |

**Total for 37 remaining files:** ~3-4 hours

---

## Need Help?

1. **Reference examples:** Check the 3 fixed files
   - `container-health-monitor.ts`
   - `database-backup-validator.ts`
   - `log-aggregation-analyzer.ts`

2. **Security utilities:** Review `security-utils.ts`
   - All functions documented
   - Usage examples in comments

3. **Common patterns:** See above patterns by file type

4. **Testing:** Use the test templates provided

---

**Remember:** Security is critical. Take your time to apply these fixes correctly. Test thoroughly before deploying to production.

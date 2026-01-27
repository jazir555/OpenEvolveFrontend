# Wave 2C Refactoring - Developer Quick Reference

**Last Updated:** January 18, 2026
**Purpose:** Quick guide for developers implementing Wave 2C refactoring

---

## 🎯 What is Wave 2C?

Wave 2C is a technical debt refactoring initiative to improve code quality across 110 BubbleLab bubble files.

**Key Goals:**
- ✅ Replace 1,128 magic numbers with named constants
- ✅ Replace 480 console.log statements with structured logging
- ✅ Extract 163 long methods (>50 lines)
- ✅ Eliminate 152+ code duplications
- ✅ Replace 210 `any` types with proper TypeScript types

---

## 📦 Shared Utilities (NEW!)

All utilities are in `/bubble-core/src/utils/`

### Importing Utilities

```typescript
// Import all utilities from central index
import {
  HTTP_TIMEOUT_DEFAULT,
  RETRY_DEFAULT_ATTEMPTS,
  createLogger,
  wrapAsync,
  createApiClient,
  validateAndParse,
} from '../utils/index.js';
```

### 1. Constants (`utils/constants.ts`)

**Commonly Used:**
```typescript
import {
  // Timeouts
  HTTP_TIMEOUT_DEFAULT,      // 30000 ms
  HTTP_TIMEOUT_SHORT,        // 5000 ms
  HTTP_TIMEOUT_LONG,         // 60000 ms

  // Retries
  RETRY_DEFAULT_ATTEMPTS,    // 3
  RETRY_MIN_DELAY_MS,        // 1000 ms

  // Pagination
  PAGE_SIZE_DEFAULT,         // 50
  PAGE_SIZE_MAX,             // 500

  // HTTP Status
  HTTP_STATUS_OK,            // 200
  HTTP_STATUS_NOT_FOUND,     // 404
  HTTP_STATUS_INTERNAL_ERROR,// 500

  // File Sizes
  MAX_FILE_SIZE_MEDIUM,      // 10 MB
  MAX_FILE_SIZE_LARGE,       // 100 MB
} from '../utils/constants.js';
```

**Before:**
```typescript
setTimeout(() => callback(), 5000);
if (response.status === 200) { ... }
```

**After:**
```typescript
setTimeout(() => callback(), HTTP_TIMEOUT_SHORT);
if (response.status === HTTP_STATUS_OK) { ... }
```

### 2. Logger (`utils/logger.ts`)

```typescript
import { createLogger } from '../utils/logger.js';

const logger = createLogger('MyBubble');

// Debug (development only)
logger.debug('Detailed debug info', { variable: value });

// Info
logger.info('Operation started', { operation_id: '123' });

// Warning
logger.warn('Unexpected condition', { condition: 'value' });

// Error
logger.error('Operation failed', error, {
  operation_id: '123',
  attempt_count: 3,
});

// With timing
const result = await logger.time('database-query', async () => {
  return await db.query('SELECT * FROM users');
});
```

**Before:**
```typescript
console.log('Processing file:', filename);
console.error('Error occurred:', error);
```

**After:**
```typescript
logger.info('Processing file', { filename });
logger.error('Processing failed', error, { filename });
```

### 3. Result Type (`utils/result.ts`)

```typescript
import { wrapAsync, ok, err, type Result } from '../utils/result.js';

// Wrap async operations
const result = await wrapAsync(async () => {
  return await riskyOperation();
});

if (result.success) {
  // Use result.data
  console.log(result.data);
} else {
  // Handle result.error
  console.error(result.error);
}

// Create results manually
return ok(data);
return err(new Error('Failed'));

// Retry with exponential backoff
const result = await retry(
  () => fetch('https://api.example.com/data'),
  { maxAttempts: 3, initialDelayMs: 1000 }
);
```

### 4. API Client (`utils/api-client.ts`)

```typescript
import { createApiClient } from '../utils/api-client.js';
import { API_ENDPOINTS } from '../config/api-endpoints.js';

// Create client
const apiClient = createApiClient({
  baseURL: API_ENDPOINTS.slack.baseURL,
  timeout: HTTP_TIMEOUT_DEFAULT,
  retryAttempts: RETRY_DEFAULT_ATTEMPTS,
  defaultHeaders: {
    'Content-Type': 'application/json',
  },
});

// Make requests
const result = await apiClient.post('/chat.postMessage', {
  channel: '#general',
  text: 'Hello, World!',
});

if (result.success) {
  const data = result.data.data;
  const status = result.data.status;
  const headers = result.data.headers;
}
```

**Authenticated API Client:**
```typescript
import { createAuthenticatedApiClient } from '../utils/api-client.js';

const apiClient = createAuthenticatedApiClient(
  { baseURL: API_ENDPOINTS.github.baseURL },
  () => getAccessToken() // Function that returns token
);

// Token is automatically added to all requests
const result = await apiClient.get('/user/repos');
```

### 5. Validation (`utils/validation.ts`)

```typescript
import { validateAndParse, safeValidate } from '../utils/validation.js';
import { z } from 'zod';

// Define schema
const UserSchema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
  age: z.number().int().positive(),
});

// Validate (throws on error)
const user = validateAndParse(UserSchema, inputData);

// Safe validate (returns Result)
const result = safeValidate(UserSchema, inputData);
if (result.success) {
  const user = result.data;
} else {
  // Handle validation errors
  console.error(result.error.errors);
}

// Common schemas
import { CommonSchemas } from '../utils/validation.js';

const email = validateAndParse(CommonSchemas.email, userInput);
const pagination = validateAndParse(CommonSchemas.pagination, queryParams);
```

---

## 🔄 Common Refactoring Patterns

### Pattern 1: Extract Long Method

**Before:**
```typescript
async processFile(file: File): Promise<Result> {
  // 150 lines of mixed logic
  const content = await readFile(file);
  const parsed = parseContent(content);
  const validated = validateData(parsed);
  const transformed = transformData(validated);
  const saved = await saveToDatabase(transformed);
  return saved;
}
```

**After:**
```typescript
async processFile(file: File): Promise<Result> {
  const content = await this.readFileContent(file);
  const parsed = this.parseFileContent(content);
  const validated = this.validateParsedData(parsed);
  const transformed = this.transformValidatedData(validated);
  return await this.saveTransformedData(transformed);
}

private async readFileContent(file: File): Promise<string> {
  logger.info('Reading file', { filename: file.name });
  return await readFile(file);
}

private parseFileContent(content: string): ParsedData {
  logger.debug('Parsing content', { content_length: content.length });
  return parseContent(content);
}

// ... more extracted methods
```

### Pattern 2: Replace Magic Numbers

**Before:**
```typescript
if (data.length > 100) {
  throw new Error('Too many items');
}

setTimeout(() => callback(), 5000);

const pageSize = 50;
```

**After:**
```typescript
import { MAX_ARRAY_SIZE_SMALL, HTTP_TIMEOUT_SHORT, PAGE_SIZE_DEFAULT } from '../utils/constants.js';

if (data.length > MAX_ARRAY_SIZE_SMALL) {
  throw new Error(`Too many items. Maximum: ${MAX_ARRAY_SIZE_SMALL}`);
}

setTimeout(() => callback(), HTTP_TIMEOUT_SHORT);

const pageSize = PAGE_SIZE_DEFAULT;
```

### Pattern 3: Simplify Conditionals

**Before:**
```typescript
if (file && file.size > 0 && file.type && (file.type.includes('pdf') || file.type.includes('document')) && (options?.validate === true || options?.strict === false)) {
  // Process file
}
```

**After:**
```typescript
if (this.isValidFileForProcessing(file, options)) {
  // Process file
}

private isValidFileForProcessing(file: File, options?: ProcessingOptions): boolean {
  const hasValidSize = file?.size > 0;
  const hasSupportedType = this.isSupportedFileType(file?.type);
  const shouldValidate = this.shouldProcessWithValidation(options);

  return hasValidSize && hasSupportedType && shouldValidate;
}

private isSupportedFileType(mimeType?: string): boolean {
  if (!mimeType) return false;
  const supportedTypes = ['pdf', 'document', 'text'];
  return supportedTypes.some(type => mimeType.includes(type));
}

private shouldProcessWithValidation(options?: ProcessingOptions): boolean {
  return options?.validate === true || options?.strict === false;
}
```

### Pattern 4: Use Guard Clauses

**Before:**
```typescript
async processRequest(request: Request) {
  if (request) {
    if (request.user) {
      if (request.user.isValid) {
        // Main logic here (nested 3 levels)
        return await this.processValidRequest(request);
      } else {
        throw new Error('Invalid user');
      }
    } else {
      throw new Error('No user');
    }
  } else {
    throw new Error('No request');
  }
}
```

**After:**
```typescript
async processRequest(request: Request) {
  // Guard clauses - fail fast
  if (!request) {
    throw new Error('No request');
  }

  if (!request.user) {
    throw new Error('No user');
  }

  if (!request.user.isValid) {
    throw new Error('Invalid user');
  }

  // Main logic - now clear and un-nested
  return await this.processValidRequest(request);
}
```

### Pattern 5: Type Safety

**Before:**
```typescript
async processData(data: any): Promise<any> {
  const result = await someOperation(data);
  return result;
}
```

**After:**
```typescript
interface ProcessInput {
  id: string;
  values: number[];
  config?: Record<string, unknown>;
}

interface ProcessOutput {
  success: boolean;
  result?: number;
  error?: string;
}

async processData(data: ProcessInput): Promise<ProcessOutput> {
  const result = await someOperation(data);
  return {
    success: true,
    result: result.value,
  };
}
```

---

## ✅ Refactoring Checklist

Before submitting a refactored file:

- [ ] All magic numbers replaced with constants
- [ ] All `console.log` replaced with `logger`
- [ ] All methods under 50 lines (ideally under 20)
- [ ] All `any` types replaced with proper types
- [ ] Complex conditionals extracted to functions
- [ ] Error handling uses Result type
- [ ] API calls use api-client
- [ ] Added JSDoc comments for public methods
- [ ] Tests added/updated
- [ ] No new magic numbers introduced
- [ ] Code reviewed by team member

---

## 📊 Progress Tracking

### Phase Status

- ✅ Phase 1: Foundation (COMPLETED)
  - ✅ Created shared utilities
  - ✅ Generated documentation
  - ✅ Built analysis tools

- 🔄 Phase 2: High-Impact Files (IN PROGRESS)
  - Refactoring top 10 files by issue count

- ⏳ Phase 3: Type Safety (PENDING)
  - Replace all `any` types

- ⏳ Phase 4: Code Deduplication (PENDING)
  - Migrate to shared utilities

- ⏳ Phase 5: Testing & Polish (PENDING)
  - Comprehensive test coverage

### Top 10 Files to Refactor

1. tool-bubble/chart-js-tool.ts (102 issues)
2. service-bubble/ai-agent.ts (86 issues)
3. tool-bubble/reddit-scrape-tool.ts (77 issues)
4. workflow-bubble/generate-document.workflow.ts (55 issues)
5. tool-bubble/pdf-generator-tool.ts (50 issues)
6. service-bubble/github.ts (49 issues)
7. service-bubble/stripe-bubble.ts (47 issues)
8. workflow-bubble/parse-document.workflow.ts (46 issues)
9. workflow-bubble/pdf-ocr.workflow.ts (44 issues)
10. service-bubble/hephaestus-bubble.ts (42 issues)

---

## 🆘 Getting Help

### Documentation

- **This Guide:** Quick reference for common patterns
- **`WAVE_2C_REFACTORING_GUIDE.md`:** Complete refactoring guide
- **`WAVE_2C_EXAMPLE_REFACTORINGS.md`:** Before/after examples
- **`WAVE_2C_TECHNICAL_DEBT_FINAL_REPORT.md`:** Full analysis report

### Tools

- **`technical_debt_analyzer.py`:** Run analysis on files
  ```bash
  python technical_debt_analyzer.py
  ```

- **`TECHNICAL_DEBT_REPORT.md`:** See issues by file

### Team Communication

- **Slack:** #wave-2c-refactoring
- **Standups:** Daily progress updates
- **Retrospective:** Weekly review and learnings

---

## 🎓 Best Practices

### DO ✅

- Write tests before refactoring (TDD)
- Keep methods under 20 lines
- Use descriptive names
- Extract constants for magic numbers
- Use Result type for error handling
- Add JSDoc comments
- Run tests after each change
- Get code reviews

### DON'T ❌

- Don't change behavior (only structure)
- Don't introduce new magic numbers
- Don't use `console.log`
- Don't create methods over 50 lines
- Don't use `any` type
- Don't skip code review
- Don't refactor without tests

---

## 📈 Metrics

Track these metrics weekly:

| Metric | Current | Target |
|--------|---------|--------|
| Magic Numbers | 1,128 | 0 |
| Console Logs | 480 | 0 |
| Long Methods | 163 | 0 |
| Type Safety | 78% | 95% |
| Code Duplication | ~4,400 lines | ~200 lines |
| Test Coverage | 45% | 80% |

---

## 🚀 Quick Start

### Refactoring Your First File

1. **Choose a file** from the top 10 list
2. **Read the analysis** in `TECHNICAL_DEBT_REPORT.md`
3. **Review examples** in `WAVE_2C_EXAMPLE_REFACTORINGS.md`
4. **Create a branch** `refactor/[filename]`
5. **Write tests** for existing behavior
6. **Apply refactoring patterns** from this guide
7. **Run tests** to ensure no behavior changes
8. **Get code review** from team
9. **Submit PR** with `Wave 2C` label
10. **Celebrate!** 🎉

---

**Remember:** The goal is to improve code quality WITHOUT changing behavior. When in doubt, ask the team!

**Questions?** Check the full documentation or ask in #wave-2c-refactoring

*Happy Refactoring! 🎯*

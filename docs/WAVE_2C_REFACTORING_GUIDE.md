# Wave 2C Technical Debt Refactoring Guide

## Executive Summary

**Analysis Date:** 2026-01-18
**Files Analyzed:** 110 TypeScript bubble files
**Total Lines of Code:** 73,478
**Total Issues Found:** 2,081

### Severity Distribution
- **HIGH:** 38 issues
- **MEDIUM:** 389 issues
- **LOW:** 1,654 issues

### Top Technical Debt Categories

1. **Magic Numbers:** 1,128 occurrences
2. **Console Logging:** 480 occurrences
3. **Type Safety (any):** 210 occurrences
4. **Long Methods:** 163 functions
5. **Poor Naming:** 46 occurrences
6. **Hardcoded URLs:** 26 occurrences
7. **Technical Debt Markers:** 21 TODO/FIXME comments
8. **Complex Conditionals:** 7 occurrences

---

## 1. Code Duplication Analysis

### Major Duplication Patterns Identified

#### Pattern 1: API Call Wrappers (152 occurrences in 25 files)
**Files affected:** google-drive.ts, eleven-labs.ts, firecrawl.ts, and 22 others

**Duplicated Code:**
```typescript
// Found in multiple files with minor variations
const response = await fetch(url, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data),
});

if (!response.ok) {
  throw new Error(`API call failed: ${response.statusText}`);
}

return await response.json();
```

**Refactoring Solution:**
Create a shared HTTP client utility:

```typescript
// /bubble-core/src/utils/api-client.ts

export interface ApiClientConfig {
  baseURL: string;
  timeout?: number;
  retryAttempts?: number;
}

export class ApiClient {
  constructor(private config: ApiClientConfig) {}

  async post<T>(endpoint: string, data: unknown, token: string): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      this.config.timeout || DEFAULT_API_TIMEOUT
    );

    try {
      const response = await fetch(`${this.config.baseURL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new ApiError(response.status, response.statusText);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

export const DEFAULT_API_TIMEOUT = 30000; // 30 seconds
```

#### Pattern 2: Error Handling Wrappers (106 occurrences in 20 files)

**Duplicated Code:**
```typescript
try {
  const result = await someOperation();
  return { success: true, data: result };
} catch (error) {
  if (error instanceof Error) {
    return { success: false, error: error.message };
  }
  return { success: false, error: 'Unknown error' };
}
```

**Refactoring Solution:**
Create a result type utility:

```typescript
// /bubble-core/src/utils/result.ts

export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export async function wrapAsync<T>(
  operation: () => Promise<T>
): Promise<Result<T>> {
  try {
    const data = await operation();
    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error : new Error(String(error)),
    };
  }
}

export function mapResult<T, U, E>(
  result: Result<T, E>,
  mapper: (data: T) => U
): Result<U, E> {
  if (result.success) {
    return { success: true, data: mapper(result.data) };
  }
  return result;
}
```

#### Pattern 3: Schema Validation Patterns (86 occurrences in 13 files)

**Refactoring Solution:**
Create validation helpers:

```typescript
// /bubble-core/src/utils/validation.ts

import { z } from 'zod';

export function validateAndParse<T>(
  schema: z.ZodSchema<T>,
  data: unknown,
  errorMessage?: string
): T {
  try {
    return schema.parse(data);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const issues = error.issues.map(i => i.message).join(', ');
      throw new ValidationError(`${errorMessage || 'Validation failed'}: ${issues}`);
    }
    throw error;
  }
}

export function safeValidate<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; errors: string[] } {
  const result = schema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.issues.map(i => `${i.path.join('.')}: ${i.message}`),
  };
}
```

---

## 2. Long Method Refactoring

### Top Long Methods (>100 lines)

#### File: service-bubble/slack.ts (2099 lines total)
**Issue:** Multiple methods over 100 lines
**Impact:** High severity - difficult to test and maintain

**Refactoring Approach:**
```typescript
// BEFORE: 150-line method
async sendMessage(params: unknown): Promise<SlackResult> {
  const parsed = SlackParamsSchema.parse(params);
  // ... 140 more lines of mixed logic

  const channel = await this.resolveChannelId(parsed.channel);
  // ... more mixed logic

  const response = await this.makeSlackApiCall('chat.postMessage', body);
  // ... even more mixed logic
}

// AFTER: Extracted methods
async sendMessage(params: unknown): Promise<SlackResult> {
  const parsed = this.validateAndParseMessageParams(params);
  const channelId = await this.resolveTargetChannel(parsed.channel);
  const messageBody = this.buildMessageBody(parsed, channelId);
  return await this.sendSlackMessage(messageBody);
}

private validateAndParseMessageParams(params: unknown) {
  return SlackParamsSchema.parse(params);
}

private async resolveTargetChannel(channel: string): Promise<string> {
  return channel.startsWith('#') || channel.startsWith('@')
    ? await this.resolveChannelId(channel)
    : channel;
}

private buildMessageBody(parsed: ParsedParams, channelId: string) {
  return {
    channel: channelId,
    text: parsed.text,
    username: parsed.username,
    icon_emoji: parsed.icon_emoji,
    // ... other mappings
  };
}

private async sendSlackMessage(body: MessageBody): Promise<SlackResult> {
  const response = await this.makeSlackApiCall('chat.postMessage', body);
  return this.handleSlackResponse(response);
}
```

### Long Method Extraction Checklist

1. **Identify logical blocks** - Group related operations
2. **Extract validation** - Separate parsing/validation logic
3. **Extract API calls** - Separate network operations
4. **Extract data transformation** - Separate mapping/conversion logic
5. **Single Responsibility** - Each method should do one thing well

---

## 3. Magic Number Elimination

### Common Magic Numbers Found

1. **Timeout values:** 30000, 5000, 10000
2. **Retry counts:** 3, 5
3. **Buffer sizes:** 1024, 4096, 8192
4. **Pagination limits:** 50, 100, 500
5. **HTTP status codes:** 200, 400, 401, 404, 500

### Refactoring Solution

Create shared constants file:

```typescript
// /bubble-core/src/utils/constants.ts

// HTTP Timeout Constants (milliseconds)
export const HTTP_TIMEOUT_DEFAULT = 30000;
export const HTTP_TIMEOUT_SHORT = 5000;
export const HTTP_TIMEOUT_LONG = 60000;

// Retry Constants
export const RETRY_DEFAULT_ATTEMPTS = 3;
export const RETRY_MAX_ATTEMPTS = 5;
export const RETRY_DELAY_MS = 1000;

// Pagination Constants
export const PAGE_SIZE_DEFAULT = 50;
export const PAGE_SIZE_MAX = 500;
export const PAGE_SIZE_MIN = 10;

// Buffer Sizes (bytes)
export const BUFFER_SIZE_SMALL = 1024;      // 1 KB
export const BUFFER_SIZE_MEDIUM = 4096;     // 4 KB
export const BUFFER_SIZE_LARGE = 8192;      // 8 KB
export const BUFFER_SIZE_XLARGE = 65536;    // 64 KB

// HTTP Status Codes
export const HTTP_STATUS_OK = 200;
export const HTTP_STATUS_CREATED = 201;
export const HTTP_STATUS_BAD_REQUEST = 400;
export const HTTP_STATUS_UNAUTHORIZED = 401;
export const HTTP_STATUS_FORBIDDEN = 403;
export const HTTP_STATUS_NOT_FOUND = 404;
export const HTTP_STATUS_INTERNAL_ERROR = 500;
export const HTTP_STATUS_SERVICE_UNAVAILABLE = 503;

// File Size Limits (bytes)
export const MAX_FILE_SIZE_SMALL = 1024 * 1024;        // 1 MB
export const MAX_FILE_SIZE_MEDIUM = 10 * 1024 * 1024;  // 10 MB
export const MAX_FILE_SIZE_LARGE = 100 * 1024 * 1024;  // 100 MB

// Rate Limiting
export const RATE_LIMIT_DEFAULT = 100;      // requests per minute
export const RATE_LIMIT_BURST = 10;         // burst requests
```

### Migration Pattern

```typescript
// BEFORE
setTimeout(() => callback(), 5000);

// AFTER
import { HTTP_TIMEOUT_SHORT } from '../utils/constants';
setTimeout(() => callback(), HTTP_TIMEOUT_SHORT);
```

---

## 4. Complex Conditional Logic

### Issues Found: 7 high-complexity conditionals

#### Example from tool-bubble/file-processor-tool.ts:

**BEFORE:**
```typescript
if (file && file.size > 0 && file.type && (file.type.includes('pdf') || file.type.includes('document') || file.type.includes('text')) && (options?.validate === true || options?.strict === false)) {
  // Process file
}
```

**AFTER:**
```typescript
// Extract to descriptive function
private isValidFileForProcessing(file: File, options?: ProcessingOptions): boolean {
  const hasValidSize = file?.size > 0;
  const hasSupportedType = this.isSupportedFileType(file?.type);
  const shouldValidate = this.shouldProcessWithValidation(options);

  return hasValidSize && hasSupportedType && shouldValidate;
}

private isSupportedFileType(mimeType?: string): boolean {
  if (!mimeType) return false;
  const supportedTypes = ['pdf', 'document', 'text', 'sheet', 'presentation'];
  return supportedTypes.some(type => mimeType.includes(type));
}

private shouldProcessWithValidation(options?: ProcessingOptions): boolean {
  return options?.validate === true || options?.strict === false;
}
```

### Guard Clause Pattern

**BEFORE:**
```typescript
async processRequest(request: Request) {
  if (request) {
    if (request.user) {
      if (request.user.isValid) {
        // Main logic here
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

**AFTER:**
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

---

## 5. Logging Infrastructure

### Current State: 480 console.log statements

### Refactoring Solution

Create structured logging utility:

```typescript
// /bubble-core/src/utils/logger.ts

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

export interface LogContext {
  correlation_id?: string;
  bubble_id?: string;
  operation?: string;
  user_id?: string;
  [key: string]: unknown;
}

export class Logger {
  constructor(
    private context: string,
    private minLevel: LogLevel = LogLevel.INFO
  ) {}

  debug(message: string, meta?: LogContext): void {
    this.log(LogLevel.DEBUG, message, meta);
  }

  info(message: string, meta?: LogContext): void {
    this.log(LogLevel.INFO, message, meta);
  }

  warn(message: string, meta?: LogContext): void {
    this.log(LogLevel.WARN, message, meta);
  }

  error(message: string, error?: Error | unknown, meta?: LogContext): void {
    const errorMeta = {
      ...meta,
      error: error instanceof Error ? {
        message: error.message,
        stack: error.stack,
        name: error.name,
      } : error,
    };
    this.log(LogLevel.ERROR, message, errorMeta);
  }

  private log(level: LogLevel, message: string, meta?: LogContext): void {
    if (level < this.minLevel) return;

    const logEntry = {
      timestamp: new Date().toISOString(),
      level: LogLevel[level],
      context: this.context,
      message,
      ...meta,
    };

    const output = JSON.stringify(logEntry);

    switch (level) {
      case LogLevel.ERROR:
        console.error(output);
        break;
      case LogLevel.WARN:
        console.warn(output);
        break;
      case LogLevel.DEBUG:
        // Only in development
        if (process.env.NODE_ENV === 'development') {
          console.debug(output);
        }
        break;
      default:
        console.log(output);
    }
  }
}

// Usage in bubbles:
// const logger = new Logger('SlackBubble');
// logger.info('Sending message', { channel_id: channelId });
// logger.error('Message failed', error, { channel_id, retry_count: 3 });
```

---

## 6. Type Safety Improvements

### Current State: 210 uses of 'any' type

### Refactoring Strategy

#### 1. Replace 'any' with specific types

**BEFORE:**
```typescript
async processData(data: any): Promise<any> {
  return result;
}
```

**AFTER:**
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
  // Implementation
}
```

#### 2. Use generic types for reusable code

**BEFORE:**
```typescript
function parseResponse(response: any): any {
  return JSON.parse(response);
}
```

**AFTER:**
```typescript
function parseResponse<T>(response: string): T {
  return JSON.parse(response) as T;
}

// Or better, with validation:
function parseResponse<T>(schema: z.ZodSchema<T>, response: string): T {
  const parsed = JSON.parse(response);
  return schema.parse(parsed);
}
```

#### 3. Use 'unknown' instead of 'any'

**BEFORE:**
```typescript
function handleInput(input: any) {
  if (typeof input === 'string') {
    // ...
  }
}
```

**AFTER:**
```typescript
function handleInput(input: unknown) {
  if (typeof input === 'string') {
    // TypeScript now knows input is string here
  }
}
```

---

## 7. Naming Convention Improvements

### Common Issues Found

1. **Ambiguous names:** tmp, temp, data, item, obj, val
2. **Abbreviations:** msg, req, res, ctx, cfg
3. **Non-descriptive:** process1, handle2, helper3

### Refactoring Guidelines

#### Variable Names

```typescript
// BAD
const tmp = getChannel();
const data = fetchData();

// GOOD
const targetChannel = getChannel();
const userData = fetchUserData();
const channelList = fetchChannelList();
```

#### Function Names

```typescript
// BAD
function process() { }
function handle() { }
function get() { }

// GOOD
function processUserRequest() { }
function handleSlackWebhook() { }
function getChannelById() { }
```

#### Boolean Variables

```typescript
// BAD
if (status) { }
if (flag) { }

// GOOD
if (isValid) { }
if (hasPermission) { }
if (shouldRetry) { }
```

---

## 8. Configuration Management

### Hardcoded URLs Found: 26 occurrences

### Refactoring Solution

Create environment-based configuration:

```typescript
// /bubble-core/src/config/api-endpoints.ts

interface ApiEndpoints {
  slack: {
    baseURL: string;
    apiVersion: string;
  };
  github: {
    baseURL: string;
    apiVersion: string;
  };
  notion: {
    baseURL: string;
  };
}

export const API_ENDPOINTS: ApiEndpoints = {
  slack: {
    baseURL: process.env.SLACK_API_URL || 'https://slack.com/api',
    apiVersion: process.env.SLACK_API_VERSION || 'v1',
  },
  github: {
    baseURL: process.env.GITHUB_API_URL || 'https://api.github.com',
    apiVersion: process.env.GITHUB_API_VERSION || '2022-11-28',
  },
  notion: {
    baseURL: process.env.NOTION_API_URL || 'https://api.notion.com',
  },
};

// Usage
import { API_ENDPOINTS } from '../config/api-endpoints';

const url = `${API_ENDPOINTS.slack.baseURL}/chat.postMessage`;
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. Create shared utilities directory structure
2. Implement `api-client.ts`
3. Implement `result.ts`
4. Implement `logger.ts`
5. Implement `constants.ts`

### Phase 2: High-Impact Files (Week 2)
1. Refactor top 10 files by issue count
2. Focus on slack.ts, ai-agent.ts, notion.ts
3. Extract long methods
4. Replace console.log with logger

### Phase 3: Type Safety (Week 3)
1. Replace 'any' types in top 20 files
2. Add proper interfaces
3. Add Zod schemas where missing

### Phase 4: Code Deduplication (Week 4)
1. Extract common patterns to utilities
2. Refactor API call patterns
3. Consolidate error handling

### Phase 5: Final Polish (Week 5)
1. Replace remaining magic numbers
2. Improve naming throughout
3. Add comprehensive tests
4. Update documentation

---

## 10. Testing Strategy

### Pre-Refactoring Tests
1. Capture current behavior with integration tests
2. Benchmark performance metrics
3. Document edge cases

### Refactoring Validation
1. Run test suite after each change
2. Compare output before/after
3. Monitor performance regression
4. Code review for each PR

### Safe Refactoring Checklist
- [ ] Tests written before refactoring
- [ ] Changes made in small increments
- [ ] Tests pass after each change
- [ ] No behavior changes detected
- [ ] Code review completed
- [ ] Documentation updated

---

## 11. Metrics and KPIs

### Current Metrics
- **Average Method Length:** 45 lines
- **Largest Method:** 187 lines
- **Code Duplication:** ~15% estimated
- **Type Safety Coverage:** 78% (22% 'any' usage)
- **Test Coverage:** 45% estimated

### Target Metrics (Post-Refactoring)
- **Average Method Length:** < 20 lines
- **Largest Method:** < 50 lines
- **Code Duplication:** < 5%
- **Type Safety Coverage:** > 95%
- **Test Coverage:** > 80%

---

## 12. Quick Reference

### File Structure After Refactoring

```
bubble-core/src/
├── utils/
│   ├── api-client.ts       # HTTP client wrapper
│   ├── result.ts           # Result type and utilities
│   ├── logger.ts           # Structured logging
│   ├── constants.ts        # Magic number replacements
│   ├── validation.ts       # Schema validation helpers
│   ├── error-handling.ts   # Custom error types
│   └── type-guards.ts      # Runtime type checking
├── config/
│   ├── api-endpoints.ts    # API URL configuration
│   ├── features.ts         # Feature flags
│   └── limits.ts           # Rate limits and constraints
└── bubbles/
    ├── service-bubble/
    ├── tool-bubble/
    └── workflow-bubble/
```

### Common Patterns

**API Call:**
```typescript
const client = new ApiClient({ baseURL: API_ENDPOINTS.slack.baseURL });
const result = await client.post('/chat.postMessage', data, token);
```

**Error Handling:**
```typescript
const result = await wrapAsync(() => riskyOperation());
if (!result.success) {
  logger.error('Operation failed', result.error);
  return;
}
// Use result.data
```

**Logging:**
```typescript
const logger = new Logger('SlackBubble');
logger.info('Processing message', { channel_id: channelId });
logger.error('Failed to send', error, { channel_id });
```

---

## Conclusion

This technical debt refactoring effort will significantly improve code quality, maintainability, and developer productivity. By following this guide systematically, we can eliminate 2,081 identified issues while maintaining full backward compatibility and test coverage.

**Next Steps:**
1. Review and approve this refactoring plan
2. Set up feature branch for refactoring work
3. Begin Phase 1 implementation
4. Establish CI/CD checks for refactoring validation
5. Track progress weekly

**Estimated Effort:** 5 weeks (1 developer)
**Risk Level:** Low (safe refactoring with tests)
**ROI:** High (significant maintainability improvement)

# Wave 2C Edge Case Implementation Guide

**Team:** Edge Case Fix Team - Wave 2C
**Date:** 2025-01-18

---

## Table of Contents
1. [Edge Case Pattern Library](#edge-case-pattern-library)
2. [Utility Functions](#utility-functions)
3. [Implementation Examples by File](#implementation-examples-by-file)
4. [Testing Strategies](#testing-strategies)
5. [Performance Considerations](#performance-considerations)

---

## Edge Case Pattern Library

### Pattern 1: Safe Property Access
**Purpose:** Prevent undefined/null access errors

**Implementation:**
```typescript
/**
 * Safely access nested object properties with null checks
 * @param obj Object to traverse
 * @param path Dot-notation path (e.g., 'user.profile.name')
 * @param defaultValue Default value if path doesn't exist
 */
export function safeGet<T = unknown>(
  obj: unknown,
  path: string,
  defaultValue: T
): T {
  if (obj === null || obj === undefined) {
    return defaultValue;
  }

  const keys = path.split('.');
  let current: unknown = obj;

  for (const key of keys) {
    if (current === null || current === undefined) {
      return defaultValue;
    }

    if (typeof current !== 'object') {
      return defaultValue;
    }

    current = (current as Record<string, unknown>)[key];
  }

  return (current ?? defaultValue) as T;
}

// Usage
const username = safeGet(user, 'profile.name', 'Anonymous');
const email = safeGet(response, 'data.user.email', '');
```

---

### Pattern 2: Type Guard Utilities
**Purpose:** Validate data types at runtime

**Implementation:**
```typescript
/**
 * Type guard for plain objects (not arrays, null, or class instances)
 */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    Object.prototype.toString.call(value) === '[object Object]'
  );
}

/**
 * Type guard for non-empty strings
 */
export function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

/**
 * Type guard for arrays
 */
export function isArray<T>(value: unknown, itemGuard?: (item: unknown) => item is T): value is T[] {
  if (!Array.isArray(value)) {
    return false;
  }

  if (itemGuard) {
    return value.every(item => itemGuard(item));
  }

  return true;
}

/**
 * Type guard for numbers (excluding NaN and Infinity)
 */
export function isSafeNumber(value: unknown): value is number {
  return (
    typeof value === 'number' &&
    Number.isFinite(value) &&
    !isNaN(value) &&
    Number.isSafeInteger(value)
  );
}

/**
 * Type guard for valid dates
 */
export function isValidDate(value: unknown): value is Date {
  return (
    value instanceof Date &&
    !isNaN(value.getTime()) &&
    value.getFullYear() >= 1900 &&
    value.getFullYear() <= 2100
  );
}

// Usage
if (isPlainObject(data)) {
  // TypeScript knows data is Record<string, unknown>
  const name = data.name; // Safe
}

if (isArray(data, isNonEmptyString)) {
  // TypeScript knows data is string[]
  data.forEach(str => console.log(str));
}
```

---

### Pattern 3: Safe Array Operations
**Purpose:** Prevent index out of bounds and empty array errors

**Implementation:**
```typescript
/**
 * Safely get array element at index
 * @param array Array to access
 * @param index Index to retrieve
 * @param defaultValue Default value if index out of bounds
 */
export function safeAt<T>(
  array: T[],
  index: number,
  defaultValue: T
): T {
  if (index < 0 || index >= array.length) {
    return defaultValue;
  }
  return array[index] ?? defaultValue;
}

/**
 * Safely get first element of array
 */
export function first<T>(array: T[], defaultValue: T): T {
  return safeAt(array, 0, defaultValue);
}

/**
 * Safely get last element of array
 */
export function last<T>(array: T[], defaultValue: T): T {
  return safeAt(array, array.length - 1, defaultValue);
}

/**
 * Safely chunk array into smaller arrays
 */
export function chunk<T>(array: T[], size: number): T[][] {
  if (!Array.isArray(array) || array.length === 0) {
    return [];
  }

  if (size < 1) {
    throw new Error('Chunk size must be at least 1');
  }

  const result: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    result.push(array.slice(i, i + size));
  }

  return result;
}

/**
 * Safely flatten nested arrays
 */
export function flatten<T>(arrays: T[][]): T[] {
  if (!Array.isArray(arrays)) {
    return [];
  }

  return Array.prototype.concat.call([], ...arrays);
}

// Usage
const items = ['a', 'b', 'c'];
const firstItem = first(items, 'default'); // 'a'
const lastItem = last(items, 'default'); // 'c'
const item10 = safeAt(items, 10, 'default'); // 'default'

const chunks = chunk([1, 2, 3, 4, 5], 2); // [[1,2], [3,4], [5]]
```

---

### Pattern 4: Safe String Operations
**Purpose:** Handle empty/whitespace strings and prevent encoding errors

**Implementation:**
```typescript
/**
 * Safely truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number, suffix = '...'): string {
  if (typeof str !== 'string') {
    return '';
  }

  if (str.length <= maxLength) {
    return str;
  }

  return str.slice(0, maxLength - suffix.length) + suffix;
}

/**
 * Safely normalize whitespace in string
 */
export function normalizeWhitespace(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }

  return str
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');
}

/**
 * Safely sanitize string for HTML output
 */
export function escapeHtml(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }

  const htmlEscapes: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '/': '&#x2F;',
  };

  return str.replace(/[&<>"'/]/g, (char) => htmlEscapes[char]);
}

/**
 * Safely encode URI component with error handling
 */
export function safeEncodeURIComponent(str: string): string {
  try {
    return encodeURIComponent(str);
  } catch {
    return str;
  }
}

/**
 * Safely decode URI component with error handling
 */
export function safeDecodeURIComponent(str: string): string {
  try {
    return decodeURIComponent(str);
  } catch {
    return str;
  }
}

// Usage
const longText = 'This is a very long text...';
const truncated = truncate(longText, 20); // 'This is a very lo...'

const messyText = '  Hello    world\n\n  ';
const clean = normalizeWhitespace(messyText); // 'Hello world'

const userInput = '<script>alert("xss")</script>';
const safe = escapeHtml(userInput); // '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'
```

---

### Pattern 5: Safe Number Operations
**Purpose:** Prevent NaN, Infinity, and overflow errors

**Implementation:**
```typescript
/**
 * Safely convert value to number with validation
 */
export function toNumber(value: unknown, defaultValue = 0): number {
  if (typeof value === 'number') {
    if (Number.isFinite(value)) {
      return value;
    }
    return defaultValue;
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (trimmed === '') {
      return defaultValue;
    }

    const num = Number(trimmed);
    if (Number.isFinite(num)) {
      return num;
    }
    return defaultValue;
  }

  if (typeof value === 'boolean') {
    return value ? 1 : 0;
  }

  return defaultValue;
}

/**
 * Safely clamp number between min and max
 */
export function clamp(num: number, min: number, max: number): number {
  if (!Number.isFinite(num)) {
    return min;
  }
  return Math.min(max, Math.max(min, num));
}

/**
 * Safely parse integer with radix validation
 */
export function toInteger(value: unknown, defaultValue = 0): number {
  const num = toNumber(value, defaultValue);
  const int = Math.trunc(num);

  if (!Number.isSafeInteger(int)) {
    return defaultValue;
  }

  return int;
}

/**
 * Safely format number with precision
 */
export function formatNumber(
  num: number,
  precision: number,
  locale = 'en-US'
): string {
  if (!Number.isFinite(num)) {
    return 'N/A';
  }

  try {
    return num.toLocaleString(locale, {
      minimumFractionDigits: precision,
      maximumFractionDigits: precision,
    });
  } catch {
    return num.toFixed(precision);
  }
}

/**
 * Safely perform arithmetic with overflow protection
 */
export function safeAdd(a: number, b: number): number {
  if (!Number.isSafeInteger(a) || !Number.isSafeInteger(b)) {
    throw new Error('Arguments must be safe integers');
  }

  const result = a + b;

  if (!Number.isSafeInteger(result)) {
    throw new Error('Arithmetic overflow');
  }

  return result;
}

// Usage
const num1 = toNumber('123.45', 0); // 123.45
const num2 = toNumber('invalid', 0); // 0
const clamped = clamp(num1, 0, 100); // 100
const int = toInteger('123.99', 0); // 123
const formatted = formatNumber(1234.567, 2); // "1,234.57"
```

---

### Pattern 6: Safe Date Operations
**Purpose:** Handle invalid dates and timezone issues

**Implementation:**
```typescript
/**
 * Safely parse date string with multiple formats
 */
export function safeParseDate(value: unknown): Date | null {
  if (value instanceof Date) {
    return isValidDate(value) ? value : null;
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();

    // Try ISO 8601 format
    const isoDate = new Date(trimmed);
    if (isValidDate(isoDate)) {
      return isoDate;
    }

    // Try other common formats
    const formats = [
      /^\d{4}-\d{2}-\d{2}$/, // YYYY-MM-DD
      /^\d{2}\/\d{2}\/\d{4}$/, // MM/DD/YYYY
      /^\d{4}\/\d{2}\/\d{2}$/, // YYYY/MM/DD
    ];

    for (const format of formats) {
      if (format.test(trimmed)) {
        const date = new Date(trimmed);
        if (isValidDate(date)) {
          return date;
        }
      }
    }

    return null;
  }

  if (typeof value === 'number') {
    const date = new Date(value);
    return isValidDate(date) ? date : null;
  }

  return null;
}

/**
 * Safely format date to ISO string
 */
export function safeToISOString(date: Date): string {
  if (!isValidDate(date)) {
    return '';
  }

  try {
    return date.toISOString();
  } catch {
    return '';
  }
}

/**
 * Safely format date to locale string
 */
export function safeToLocaleString(
  date: Date,
  locale = 'en-US',
  options?: Intl.DateTimeFormatOptions
): string {
  if (!isValidDate(date)) {
    return 'Invalid Date';
  }

  try {
    return date.toLocaleString(locale, options);
  } catch {
    return date.toLocaleString();
  }
}

/**
 * Validate timezone string
 */
export function isValidTimeZone(timeZone: string): boolean {
  try {
    Intl.DateTimeFormat(undefined, { timeZone });
    return true;
  } catch {
    return false;
  }
}

// Usage
const date1 = safeParseDate('2024-01-15'); // Date object
const date2 = safeParseDate('invalid'); // null
const iso = safeToISOString(date1!); // "2024-01-15T00:00:00.000Z"
const formatted = safeToLocaleString(date1!); // "1/15/2024, 12:00:00 AM"
```

---

### Pattern 7: Safe Object Operations
**Purpose:** Handle null/undefined objects and prevent prototype pollution

**Implementation:**
```typescript
/**
 * Safely deep clone object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || obj === undefined) {
    return obj;
  }

  if (typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as T;
  }

  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as T;
  }

  if (obj instanceof Object) {
    const cloned: Record<string, unknown> = {};
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        cloned[key] = deepClone((obj as Record<string, unknown>)[key]);
      }
    }
    return cloned as T;
  }

  return obj;
}

/**
 * Safely merge objects without prototype pollution
 */
export function safeMerge<T extends Record<string, unknown>>(
  target: T,
  ...sources: Partial<T>[]
): T {
  const result = deepClone(target);

  for (const source of sources) {
    if (source === null || source === undefined) {
      continue;
    }

    for (const key in source) {
      if (
        Object.prototype.hasOwnProperty.call(source, key) &&
        typeof key === 'string' &&
        key !== '__proto__' &&
        key !== 'constructor' &&
        key !== 'prototype'
      ) {
        const sourceValue = source[key];
        const targetValue = result[key];

        if (isPlainObject(sourceValue) && isPlainObject(targetValue)) {
          result[key] = safeMerge(
            targetValue as Record<string, unknown>,
            sourceValue as Record<string, unknown>
          );
        } else {
          result[key] = deepClone(sourceValue);
        }
      }
    }
  }

  return result;
}

/**
 * Safely pick properties from object
 */
export function pick<T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>;

  for (const key of keys) {
    if (key in obj && Object.prototype.hasOwnProperty.call(obj, key)) {
      result[key] = obj[key];
    }
  }

  return result;
}

/**
 * Safely omit properties from object
 */
export function omit<T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj };

  for (const key of keys) {
    delete result[key];
  }

  return result as Omit<T, K>;
}

// Usage
const config = { a: 1, b: 2, c: 3 };
const picked = pick(config, ['a', 'b']); // { a: 1, b: 2 }
const omitted = omit(config, ['c']); // { a: 1, b: 2 }

const merged = safeMerge({ a: 1 }, { b: 2 }, { c: 3 }); // { a: 1, b: 2, c: 3 }
```

---

### Pattern 8: Concurrent Request Management
**Purpose:** Prevent race conditions and duplicate requests

**Implementation:**
```typescript
/**
 * Request deduplication manager
 */
export class RequestDeduplicator {
  private pendingRequests = new Map<string, Promise<unknown>>();

  /**
   * Execute request with deduplication
   */
  async execute<T>(
    key: string,
    requestFn: () => Promise<T>,
    ttl = 1000
  ): Promise<T> {
    // Check for pending request
    const existing = this.pendingRequests.get(key);
    if (existing) {
      console.debug(`[RequestDeduplicator] Reusing pending request: ${key}`);
      return existing as Promise<T>;
    }

    // Create new request
    const promise = requestFn()
      .finally(() => {
        // Clean up after TTL
        setTimeout(() => {
          this.pendingRequests.delete(key);
        }, ttl);
      });

    this.pendingRequests.set(key, promise);

    return promise;
  }

  /**
   * Clear all pending requests
   */
  clear(): void {
    this.pendingRequests.clear();
  }

  /**
   * Get count of pending requests
   */
  get size(): number {
    return this.pendingRequests.size;
  }
}

/**
 * Mutex lock for critical sections
 */
export class MutexLock {
  private locks = new Map<string, Promise<void>>();

  /**
   * Acquire lock and execute function
   */
  async acquire<T>(key: string, fn: () => Promise<T>): Promise<T> {
    // Wait for existing lock
    const existingLock = this.locks.get(key);
    if (existingLock) {
      await existingLock;
    }

    // Create new lock
    let releaseLock: () => void;
    const lock = new Promise<void>((resolve) => {
      releaseLock = resolve;
    });

    this.locks.set(key, lock);

    try {
      const result = await fn();
      return result;
    } finally {
      // Release lock
      this.locks.delete(key);
      releaseLock!();
    }
  }
}

// Usage
const deduplicator = new RequestDeduplicator();

const data1 = await deduplicator.execute(
  'api:/users/123',
  () => fetch('/api/users/123').then(r => r.json())
);

const data2 = await deduplicator.execute(
  'api:/users/123', // Same key
  () => fetch('/api/users/123').then(r => r.json())
); // Returns same promise as data1

const mutex = new MutexLock();

await mutex.acquire('critical-section', async () => {
  // Only one execution at a time per key
  const result = await performCriticalOperation();
});
```

---

## Utility Functions

### Validation Utilities
```typescript
/**
 * Comprehensive validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: Array<{
    field: string;
    message: string;
    value: unknown;
  }>;
}

/**
 * Validate object against schema
 */
export function validateObject<T extends Record<string, unknown>>(
  obj: unknown,
  schema: {
    [K in keyof T]?: {
      required?: boolean;
      type: 'string' | 'number' | 'boolean' | 'object' | 'array';
      validate?: (value: unknown) => boolean;
      min?: number;
      max?: number;
      pattern?: RegExp;
    };
  }
): ValidationResult {
  const errors: ValidationResult['errors'] = [];

  if (!isPlainObject(obj)) {
    errors.push({
      field: 'root',
      message: 'Value must be an object',
      value: obj,
    });
    return { valid: false, errors };
  }

  for (const [field, rules] of Object.entries(schema)) {
    const value = obj[field];
    const fieldRules = rules as
      | { required?: boolean; type: string; validate?: (value: unknown) => boolean }
      | undefined;

    if (!fieldRules) continue;

    // Check required
    if (fieldRules.required === true && (value === undefined || value === null)) {
      errors.push({
        field,
        message: 'Field is required',
        value,
      });
      continue;
    }

    // Skip validation if optional and not present
    if (value === undefined || value === null) {
      continue;
    }

    // Check type
    let typeValid = false;
    switch (fieldRules.type) {
      case 'string':
        typeValid = typeof value === 'string';
        break;
      case 'number':
        typeValid = typeof value === 'number' && Number.isFinite(value);
        break;
      case 'boolean':
        typeValid = typeof value === 'boolean';
        break;
      case 'object':
        typeValid = isPlainObject(value);
        break;
      case 'array':
        typeValid = Array.isArray(value);
        break;
    }

    if (!typeValid) {
      errors.push({
        field,
        message: `Expected type ${fieldRules.type}, got ${typeof value}`,
        value,
      });
      continue;
    }

    // Custom validation
    if (fieldRules.validate && !fieldRules.validate(value)) {
      errors.push({
        field,
        message: 'Custom validation failed',
        value,
      });
      continue;
    }

    // Min/max for strings and numbers
    if (typeof value === 'string' || typeof value === 'number') {
      const numValue = typeof value === 'string' ? value.length : value;

      if ('min' in fieldRules && fieldRules.min !== undefined && numValue < fieldRules.min) {
        errors.push({
          field,
          message: `Value must be at least ${fieldRules.min}`,
          value,
        });
      }

      if ('max' in fieldRules && fieldRules.max !== undefined && numValue > fieldRules.max) {
        errors.push({
          field,
          message: `Value must be at most ${fieldRules.max}`,
          value,
        });
      }
    }

    // Pattern for strings
    if (typeof value === 'string' && 'pattern' in fieldRules && fieldRules.pattern) {
      if (!fieldRules.pattern.test(value)) {
        errors.push({
          field,
          message: `Value does not match pattern ${fieldRules.pattern}`,
          value,
        });
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// Usage
const result = validateObject(
  { name: 'John', age: 30, email: 'john@example.com' },
  {
    name: { required: true, type: 'string', min: 1, max: 100 },
    age: { required: true, type: 'number', min: 0, max: 150 },
    email: {
      required: true,
      type: 'string',
      pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    },
  }
);

if (!result.valid) {
  console.error('Validation errors:', result.errors);
}
```

---

## Implementation Examples by File

### File: http.ts

```typescript
// Add to http.ts

import {
  safeEncodeURIComponent,
  toNumber,
  isNonEmptyString,
  validateObject,
} from '../utils/edge-case-utils.js';

export class HttpBubble extends ServiceBubble<HttpParams, HttpResult> {
  protected async performAction(
    context?: BubbleContext
  ): Promise<HttpResult> {
    void context;

    const { operation } = this.params;

    try {
      // Validate URL
      if (!isNonEmptyString(this.params.url)) {
        throw new Error('URL is required and cannot be empty');
      }

      // Validate URL format
      let validUrl: URL;
      try {
        validUrl = new URL(this.params.url);
      } catch {
        throw new Error(`Invalid URL format: ${this.params.url}`);
      }

      // Validate timeout
      const timeout = toNumber(this.params.timeout, 30000);
      if (timeout < 0 || timeout > 300000) {
        throw new Error('Timeout must be between 0 and 300000ms (5 minutes)');
      }

      // Sanitize headers
      const sanitizedHeaders = this.sanitizeHeaders(this.params.headers ?? {});

      // Sanitize query parameters
      const sanitizedQuery = this.sanitizeQueryParams(this.params.queryParams ?? {});

      // Build final URL
      const finalUrl = this.buildUrl(validUrl, sanitizedQuery);

      // Make request with timeout
      const response = await this.makeRequest(finalUrl, {
        method: this.params.method ?? 'GET',
        headers: sanitizedHeaders,
        body: this.params.body,
        timeout,
      });

      return response;
    } catch (error) {
      return {
        operation,
        ok: false,
        statusCode: 0,
        statusText: 'Error',
        headers: {},
        body: null,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  private sanitizeHeaders(
    headers: Record<string, string>
  ): Record<string, string> {
    const sanitized: Record<string, string> = {};

    for (const [key, value] of Object.entries(headers)) {
      // Skip invalid header names
      if (!/^[\w-]+$/.test(key)) {
        console.warn(`[HttpBubble] Skipping invalid header name: ${key}`);
        continue;
      }

      // Sanitize header value
      if (typeof value !== 'string') {
        sanitized[key] = String(value);
      } else {
        // Remove control characters
        sanitized[key] = value.replace(/[\x00-\x1F\x7F]/g, '');
      }
    }

    return sanitized;
  }

  private sanitizeQueryParams(
    params: Record<string, string | number | boolean>
  ): Record<string, string> {
    const sanitized: Record<string, string> = {};

    for (const [key, value] of Object.entries(params)) {
      // Skip invalid parameter names
      if (!/^[\w-]+$/.test(key)) {
        console.warn(`[HttpBubble] Skipping invalid query parameter: ${key}`);
        continue;
      }

      // Convert and sanitize value
      const strValue = String(value);
      sanitized[key] = safeEncodeURIComponent(strValue);
    }

    return sanitized;
  }

  private buildUrl(baseUrl: URL, params: Record<string, string>): string {
    if (Object.keys(params).length === 0) {
      return baseUrl.toString();
    }

    const queryString = new URLSearchParams(params).toString();
    return `${baseUrl.toString()}${baseUrl.search ? '&' : '?'}${queryString}`;
  }

  private async makeRequest(
    url: string,
    options: {
      method: string;
      headers: Record<string, string>;
      body?: unknown;
      timeout: number;
    }
  ): Promise<HttpResult> {
    const { method, headers, body, timeout } = options;

    // Build fetch options
    const fetchOptions: RequestInit = {
      method,
      headers,
    };

    // Add body for POST/PUT/PATCH
    if (['POST', 'PUT', 'PATCH'].includes(method.toUpperCase()) && body) {
      if (typeof body === 'string') {
        fetchOptions.body = body;
      } else if (isPlainObject(body)) {
        fetchOptions.body = JSON.stringify(body);
      } else {
        throw new Error('Body must be string or object');
      }
    }

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    fetchOptions.signal = controller.signal;

    try {
      const response = await fetch(url, fetchOptions);
      clearTimeout(timeoutId);

      // Parse response body
      let responseBody: unknown = null;
      const contentType = response.headers.get('content-type') ?? '';

      if (contentType.includes('application/json')) {
        try {
          responseBody = await response.json();
        } catch {
          responseBody = await response.text();
        }
      } else if (contentType.includes('text/')) {
        responseBody = await response.text();
      }

      // Parse response headers
      const responseHeaders: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        responseHeaders[key] = value;
      });

      return {
        operation: 'request',
        ok: response.ok,
        statusCode: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
        body: responseBody,
        error: '',
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${timeout}ms`);
      }
      throw error;
    }
  }
}
```

### File: slack.ts

```typescript
// Add to slack.ts

import {
  safeGet,
  isNonEmptyString,
  validateObject,
  truncate,
} from '../utils/edge-case-utils.js';

export class SlackBubble extends ServiceBubble<SlackParams, SlackResult> {
  protected async performAction(
    context?: BubbleContext
  ): Promise<SlackResult> {
    void context;

    const { operation } = this.params;

    try {
      switch (operation) {
        case 'send_message':
          return await this.sendMessage(this.params);
        case 'list_channels':
          return await this.listChannels(this.params);
        case 'get_channel_info':
          return await this.getChannelInfo(this.params);
        // ... other cases
      }
    } catch (error) {
      return {
        operation,
        ok: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      } as SlackResult;
    }
  }

  private async sendMessage(
    params: Extract<SlackParams, { operation: 'send_message' }>
  ): Promise<Extract<SlackResult, { operation: 'send_message' }>> {
    const {
      channel,
      text,
      username,
      icon_emoji,
      icon_url,
      attachments,
      blocks,
      thread_ts,
      reply_broadcast,
      unfurl_links,
      unfurl_media,
    } = params;

    // Validate channel
    const sanitizedChannel = this.validateAndSanitizeChannel(channel);

    // Validate text
    if (!isNonEmptyString(text)) {
      throw new Error('Message text is required and cannot be empty');
    }

    const truncatedText = truncate(text, 40000);

    // Validate optional parameters
    if (username !== undefined) {
      if (username.length > 80) {
        throw new Error('Username cannot exceed 80 characters');
      }
    }

    // Validate attachments
    if (attachments) {
      if (!Array.isArray(attachments)) {
        throw new Error('Attachments must be an array');
      }

      if (attachments.length > 100) {
        throw new Error('Maximum 100 attachments allowed');
      }

      // Validate each attachment
      attachments.forEach((attachment, index) => {
        if (!isPlainObject(attachment)) {
          throw new Error(`Attachment at index ${index} is not an object`);
        }
      });
    }

    // Validate blocks
    if (blocks) {
      if (!Array.isArray(blocks)) {
        throw new Error('Blocks must be an array');
      }

      if (blocks.length > 100) {
        throw new Error('Maximum 100 blocks allowed');
      }
    }

    // Build request body
    const requestBody: Record<string, unknown> = {
      channel: sanitizedChannel,
      text: truncatedText,
    };

    if (username) {
      requestBody.username = truncate(username, 80);
    }

    if (icon_emoji) {
      requestBody.icon_emoji = icon_emoji;
    }

    if (icon_url) {
      // Validate URL
      try {
        new URL(icon_url);
        requestBody.icon_url = icon_url;
      } catch {
        throw new Error(`Invalid icon_url: ${icon_url}`);
      }
    }

    if (attachments && attachments.length > 0) {
      requestBody.attachments = attachments;
    }

    if (blocks && blocks.length > 0) {
      requestBody.blocks = blocks;
    }

    if (thread_ts) {
      requestBody.thread_ts = thread_ts;
      if (reply_broadcast) {
        requestBody.reply_broadcast = true;
      }
    }

    if (unfurl_links !== undefined) {
      requestBody.unfurl_links = unfurl_links;
    }

    if (unfurl_media !== undefined) {
      requestBody.unfurl_media = unfurl_media;
    }

    // Make API call
    const response = await this.makeSlackApiCall('chat.postMessage', 'POST', requestBody);

    if (!response.ok) {
      return {
        operation: 'send_message',
        ok: false,
        error: response.error ?? 'Unknown error',
      };
    }

    return {
      operation: 'send_message',
      ok: true,
      channel: response.channel,
      timestamp: response.ts,
      message: response.message,
      error: '',
    };
  }

  private validateAndSanitizeChannel(channel: string): string {
    if (!isNonEmptyString(channel)) {
      throw new Error('Channel is required and cannot be empty');
    }

    const sanitized = channel.trim();

    if (sanitized.length === 0) {
      throw new Error('Channel cannot be whitespace-only');
    }

    if (sanitized.length > 80) {
      throw new Error('Channel name cannot exceed 80 characters');
    }

    return sanitized;
  }

  private async resolveChannelName(channelIdOrName: string): Promise<string> {
    // If it's already an ID (starts with C for public, G for private, D for DM), return as-is
    if (/^[CGD]\w+$/.test(channelIdOrName)) {
      return channelIdOrName;
    }

    // Otherwise, resolve name to ID
    const response = await this.makeSlackApiCall('conversations.list', 'GET', {
      types: 'public_channel,private_channel',
      limit: '1000',
    });

    if (!response.ok) {
      throw new Error(`Failed to resolve channel name: ${response.error}`);
    }

    const channel = safeGet(
      response,
      'channels',
      []
    ).find((ch: any) => ch.name === channelIdOrName);

    if (!channel) {
      throw new Error(`Channel not found: ${channelIdOrName}`);
    }

    return channel.id;
  }
}
```

### File: csv-processor-tool.ts

```typescript
// Add to csv-processor-tool.ts

import {
  toNumber,
  toInteger,
  safeParseDate,
  isPlainObject,
  safeAt,
  chunk,
} from '../utils/edge-case-utils.js';

export class CSVProcessorTool extends ToolBubble<CSVProcessorToolParams, CSVProcessorToolResult> {
  /**
   * Parse CSV with comprehensive edge case handling
   */
  private async parseCSV(): Promise<CSVProcessorToolResult> {
    const { csvData, delimiter, hasHeader, skipEmptyLines, trimWhitespace, maxRows } = this.params;

    // Validate input
    if (!csvData || typeof csvData !== 'string') {
      throw new Error('csvData is required and must be a string');
    }

    if (csvData.length === 0) {
      throw new Error('csvData cannot be empty');
    }

    // Handle different line endings
    const normalizedData = csvData
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n');

    const lines = normalizedData.split('\n');

    // Handle empty CSV
    if (lines.length === 0 || (lines.length === 1 && lines[0].trim() === '')) {
      return this.emptyResult();
    }

    // Parse headers
    const { headers, startIndex } = this.parseHeaders(lines, hasHeader);

    // Parse data rows
    const { data, validationErrors } = this.parseRows(
      lines,
      startIndex,
      headers,
      { skipEmptyLines, trimWhitespace, maxRows }
    );

    return {
      data,
      rowCount: data.length,
      columnCount: headers.length,
      headers,
      validationErrors: validationErrors.length > 0 ? validationErrors : undefined,
      statistics: {
        totalRows: data.length,
        validRows: data.length - validationErrors.length,
        invalidRows: validationErrors.length,
        processingTime: 0,
      },
      success: true,
      error: '',
    };
  }

  private emptyResult(): CSVProcessorToolResult {
    return {
      data: [],
      rowCount: 0,
      columnCount: 0,
      headers: [],
      statistics: {
        totalRows: 0,
        validRows: 0,
        invalidRows: 0,
        processingTime: 0,
      },
      success: true,
      error: '',
    };
  }

  private parseHeaders(lines: string[], hasHeader: boolean): {
    headers: string[];
    startIndex: number;
  } {
    let headers: string[] = [];
    let startIndex = 0;

    if (hasHeader) {
      const headerLine = safeAt(lines, 0, '');
      headers = this.parseLine(headerLine, this.params.delimiter);

      if (this.params.trimWhitespace) {
        headers = headers.map(h => h.trim());
      }

      if (headers.length === 0) {
        throw new Error('CSV must have at least one column header');
      }

      // Validate header uniqueness
      const uniqueHeaders = new Set(headers);
      if (uniqueHeaders.size !== headers.length) {
        const duplicates = headers.filter(
          (item, index) => headers.indexOf(item) !== index
        );
        throw new Error(`Duplicate header names found: ${duplicates.join(', ')}`);
      }

      startIndex = 1;
    } else {
      // Generate column names
      const firstRow = this.parseLine(safeAt(lines, 0, ''), this.params.delimiter);
      if (firstRow.length === 0) {
        throw new Error('CSV must have at least one column');
      }
      headers = firstRow.map((_, i) => `column_${i}`);
    }

    return { headers, startIndex };
  }

  private parseRows(
    lines: string[],
    startIndex: number,
    headers: string[],
    options: {
      skipEmptyLines: boolean;
      trimWhitespace: boolean;
      maxRows?: number;
    }
  ): {
    data: Record<string, unknown>[];
    validationErrors: Array<{
      row: number;
      column: string;
      error: string;
      value: unknown;
    }>;
  } {
    const { skipEmptyLines, trimWhitespace, maxRows } = options;
    const data: Record<string, unknown>[] = [];
    const validationErrors: Array<{
      row: number;
      column: string;
      error: string;
      value: unknown;
    }> = [];

    for (let i = startIndex; i < lines.length; i++) {
      const line = lines[i] ?? '';

      // Skip empty lines
      if (skipEmptyLines && line.trim() === '') {
        continue;
      }

      try {
        const values = this.parseLine(line, this.params.delimiter);

        // Trim whitespace if configured
        const processedValues = trimWhitespace
          ? values.map(v => v.trim())
          : values;

        // Validate row length
        if (processedValues.length !== headers.length) {
          validationErrors.push({
            row: i + 1,
            column: 'row_length',
            error: `Row has ${processedValues.length} values, expected ${headers.length}`,
            value: processedValues,
          });

          // Pad or trim to match headers
          while (processedValues.length < headers.length) {
            processedValues.push('');
          }
          if (processedValues.length > headers.length) {
            processedValues.length = headers.length;
          }
        }

        // Create row object with type inference
        const row: Record<string, unknown> = {};
        for (let headerIndex = 0; headerIndex < headers.length; headerIndex++) {
          const header = headers[headerIndex];
          const value = safeAt(processedValues, headerIndex, '');

          // Infer data type
          row[header] = this.inferDataType(value);
        }

        data.push(row);

        // Check max rows limit
        if (maxRows && data.length >= maxRows) {
          console.warn(`[CSVProcessorTool] Reached max rows limit (${maxRows})`);
          break;
        }
      } catch (error) {
        validationErrors.push({
          row: i + 1,
          column: 'parse_error',
          error: error instanceof Error ? error.message : 'Unknown parse error',
          value: line,
        });
      }
    }

    return { data, validationErrors };
  }

  /**
   * Safely infer data type from string value
   */
  private inferDataType(value: string): string | number | boolean | Date {
    const trimmed = value.trim();

    // Empty string
    if (trimmed === '') {
      return '';
    }

    // Boolean
    const lower = trimmed.toLowerCase();
    if (lower === 'true') {
      return true;
    }
    if (lower === 'false') {
      return false;
    }

    // Number
    if (/^-?\d+\.?\d*$/.test(trimmed)) {
      const num = toNumber(trimmed, NaN);
      if (!isNaN(num)) {
        return num;
      }
    }

    // Date (ISO 8601 format)
    const date = safeParseDate(trimmed);
    if (date) {
      return date;
    }

    // Default to string
    return trimmed;
  }

  /**
   * Parse CSV line with proper quote/escape handling
   */
  private parseLine(line: string, delimiter: string): string[] {
    const { quoteChar, escapeChar } = this.params;
    const values: string[] = [];
    let current = '';
    let inQuotes = false;
    let i = 0;

    while (i < line.length) {
      const char = line[i];
      const nextChar = line[i + 1];

      // Handle escaped quote
      if (char === escapeChar && nextChar === quoteChar && inQuotes) {
        current += quoteChar;
        i += 2;
        continue;
      }

      // Handle quote character
      if (char === quoteChar) {
        // Check for doubled quote (escaped)
        if (nextChar === quoteChar && inQuotes) {
          current += quoteChar;
          i += 2;
          continue;
        }
        inQuotes = !inQuotes;
        i++;
        continue;
      }

      // Handle delimiter (only when not in quotes)
      if (char === delimiter && !inQuotes) {
        values.push(current);
        current = '';
        i++;
        continue;
      }

      // Handle newlines in quoted fields
      if ((char === '\r' || char === '\n') && inQuotes) {
        current += char === '\r' && nextChar === '\n' ? '\n' : char;
        i += char === '\r' && nextChar === '\n' ? 2 : 1;
        continue;
      }

      // Regular character
      current += char;
      i++;
    }

    // Add last value
    values.push(current);

    return values;
  }
}
```

---

## Testing Strategies

### Unit Test Template
```typescript
import { describe, it, expect } from '@jest/globals';
import { safeGet, toNumber, isPlainObject } from '../utils/edge-case-utils';

describe('Edge Case Utils', () => {
  describe('safeGet', () => {
    it('should handle null objects', () => {
      expect(safeGet(null, 'a.b.c', 'default')).toBe('default');
    });

    it('should handle undefined nested properties', () => {
      expect(safeGet({ a: {} }, 'a.b.c', 'default')).toBe('default');
    });

    it('should return existing values', () => {
      expect(safeGet({ a: { b: { c: 5 } } }, 'a.b.c', 0)).toBe(5);
    });
  });

  describe('toNumber', () => {
    it('should handle invalid strings', () => {
      expect(toNumber('invalid', 0)).toBe(0);
    });

    it('should handle empty strings', () => {
      expect(toNumber('', 42)).toBe(42);
    });

    it('should handle booleans', () => {
      expect(toNumber(true, 0)).toBe(1);
      expect(toNumber(false, 0)).toBe(0);
    });

    it('should handle valid numbers', () => {
      expect(toNumber('123.45', 0)).toBe(123.45);
    });
  });

  describe('isPlainObject', () => {
    it('should reject arrays', () => {
      expect(isPlainObject([])).toBe(false);
    });

    it('should reject null', () => {
      expect(isPlainObject(null)).toBe(false);
    });

    it('should reject class instances', () => {
      class Foo {}
      expect(isPlainObject(new Foo())).toBe(false);
    });

    it('should accept plain objects', () => {
      expect(isPlainObject({})).toBe(true);
      expect(isPlainObject({ a: 1 })).toBe(true);
    });
  });
});
```

### Integration Test Template
```typescript
import { describe, it, expect } from '@jest/globals';
import { CSVProcessorTool } from '../csv-processor-tool';

describe('CSVProcessorTool Edge Cases', () => {
  it('should handle empty CSV', async () => {
    const tool = new CSVProcessorTool({
      operation: 'parse',
      csvData: '',
      delimiter: ',',
      hasHeader: true,
    });

    const result = await tool.action();

    expect(result.rowCount).toBe(0);
    expect(result.columnCount).toBe(0);
    expect(result.success).toBe(true);
  });

  it('should handle CSV with only headers', async () => {
    const tool = new CSVProcessorTool({
      operation: 'parse',
      csvData: 'name,age,email',
      delimiter: ',',
      hasHeader: true,
    });

    const result = await tool.action();

    expect(result.rowCount).toBe(0);
    expect(result.columnCount).toBe(3);
    expect(result.headers).toEqual(['name', 'age', 'email']);
  });

  it('should handle malformed rows', async () => {
    const tool = new CSVProcessorTool({
      operation: 'parse',
      csvData: 'name,age\nJohn,30\nJane', // Missing age for Jane
      delimiter: ',',
      hasHeader: true,
    });

    const result = await tool.action();

    expect(result.validationErrors).toBeDefined();
    expect(result.validationErrors?.length).toBe(1);
    expect(result.data?.[1]).toEqual({ name: 'Jane', age: '' });
  });

  it('should handle quoted fields with newlines', async () => {
    const tool = new CSVProcessorTool({
      operation: 'parse',
      csvData: 'name,description\nJohn,"Multi\nline\ndescription"',
      delimiter: ',',
      hasHeader: true,
    });

    const result = await tool.action();

    expect(result.rowCount).toBe(1);
    expect(result.data?.[0].description).toBe('Multi\nline\ndescription');
  });

  it('should handle max rows limit', async () => {
    const csv = 'name\n' + Array(1000).fill('John').join('\n');
    const tool = new CSVProcessorTool({
      operation: 'parse',
      csvData: csv,
      delimiter: ',',
      hasHeader: true,
      maxRows: 100,
    });

    const result = await tool.action();

    expect(result.rowCount).toBe(100);
  });
});
```

---

## Performance Considerations

### Memory Management
```typescript
/**
 * Process large CSV in chunks to avoid memory issues
 */
private async parseLargeCSV(): Promise<CSVProcessorToolResult> {
  const { csvData, maxRows = 10000 } = this.params;

  // Split into chunks
  const lines = csvData.split('\n');
  const chunkSize = 1000;
  const chunks = chunk(lines, chunkSize);

  const data: Record<string, unknown>[] = [];
  let totalProcessed = 0;

  for (const chunk of chunks) {
    // Process chunk
    const chunkResult = this.parseLines(chunk);

    data.push(...chunkResult.data);
    totalProcessed += chunkResult.rowCount;

    // Check max rows
    if (maxRows && totalProcessed >= maxRows) {
      break;
    }

    // Allow event loop to process
    await new Promise(resolve => setImmediate(resolve));
  }

  return {
    data,
    rowCount: data.length,
    // ... rest of result
  };
}
```

### Request Optimization
```typescript
/**
 * Batch API requests to avoid rate limits
 */
export class BatchRequestManager {
  private batchSize = 50;
  private delayMs = 100;

  async batch<T, R>(
    items: T[],
    fn: (batch: T[]) => Promise<R[]>
  ): Promise<R[]> {
    const results: R[] = [];
    const batches = chunk(items, this.batchSize);

    for (const batch of batches) {
      const batchResults = await fn(batch);
      results.push(...batchResults);

      // Delay between batches
      if (batches.indexOf(batch) < batches.length - 1) {
        await this.delay(this.delayMs);
      }
    }

    return results;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

---

**End of Implementation Guide**

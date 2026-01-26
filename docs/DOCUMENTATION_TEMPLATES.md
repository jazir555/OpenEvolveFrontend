# BubbleLab Documentation Templates & Style Guide

**Version:** 1.0.0
**Last Updated:** 2025-01-18

## Table of Contents

1. [File Templates](#file-templates)
2. [JSDoc Tag Reference](#jsdoc-tag-reference)
3. [Inline Comment Guidelines](#inline-comment-guidelines)
4. [Type Documentation](#type-documentation)
5. [Security Documentation](#security-documentation)
6. [Example Documentation](#example-documentation)

---

## File Templates

### Template 1: Service Bubble (Complete)

```typescript
/**
 * BUBBLE_NAME - Brief Description
 *
 * @module bubbles/service-bubble/bubble-name
 * @description
 * Comprehensive description of what this service bubble does.
 *
 * ## Features
 * - Feature one with details
 * - Feature two with details
 * - Feature three with details
 *
 * ## Authentication
 * Requires {@link CredentialType.CREDENTIAL_TYPE} credential.
 * Get your API key from: https://service.com/api-keys
 *
 * ## Rate Limits
 * - Free tier: 100 requests/minute
 * - Paid tier: 1000 requests/minute
 *
 * ## Common Use Cases
 * ### Use Case 1
 * ```typescript
 * const bubble = new BubbleName({
 *   param: 'value',
 *   credentials: { [CredentialType.CREDENTIAL_TYPE]: 'api-key' }
 * });
 * const result = await bubble.action();
 * ```
 *
 * ### Use Case 2
 * ```typescript
 * // Advanced usage example
 * ```
 *
 * ## Error Handling
 * - `ValidationError`: Invalid input parameters
 * - `AuthenticationError`: Invalid or missing credentials
 * - `RateLimitError`: Exceeded API rate limit
 * - `NetworkError`: Network connectivity issues
 *
 * @see [External Documentation](https://service.com/docs)
 * @author BubbleLab Team
 * @version 1.0.0
 * @license MIT
 *
 * @example
 * Basic usage:
 * ```typescript
 * import { BubbleName } from '@bubblelab/bubble-core';
 *
 * const bubble = new BubbleName({
 *   operation: 'do_something',
 *   input: 'value',
 *   credentials: {
 *     [CredentialType.CREDENTIAL_TYPE]: process.env.API_KEY
 *   }
 * });
 *
 * const result = await bubble.action();
 * if (result.success) {
 *   console.log(result.data);
 * } else {
 *   console.error(result.error);
 * }
 * ```
 */

import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';

/**
 * Input parameters for BubbleName operations.
 *
 * @property operation - The specific operation to perform
 * @property input - Description of input parameter
 * @property options - Optional configuration object
 */
export type BubbleNameParams = z.infer<typeof BubbleNameParamsSchema>;

/**
 * Result type for BubbleName operations.
 *
 * @property success - Whether the operation completed successfully
 * @property data - The returned data (present when success is true)
 * @property error - Error message (present when success is false)
 * @property metadata - Additional metadata about the operation
 */
export type BubbleNameResult = z.infer<typeof BubbleNameResultSchema>;

export class BubbleNameBubble extends ServiceBubble<BubbleNameParams, BubbleNameResult> {
  /**
   * Service identifier (used for credential lookup)
   */
  static readonly service = 'service-name';

  /**
   * Authentication type required by this service
   */
  static readonly authType = 'apikey' as const;

  /**
   * Unique bubble identifier
   */
  static readonly bubbleName = 'bubble-name';

  /**
   * Bubble type classification
   */
  static readonly type = 'service' as const;

  /**
   * Zod schema for input validation
   */
  static readonly schema = BubbleNameParamsSchema;

  /**
   * Zod schema for result validation
   */
  static readonly resultSchema = BubbleNameResultSchema;

  /**
   * Short description for UI displays
   */
  static readonly shortDescription = 'Brief one-line description';

  /**
   * Detailed description with usage examples
   */
  static readonly longDescription = `
    Extended description of what this bubble does and when to use it.

    Includes:
    - Feature list
    - Use cases
    - Configuration options
    - Common patterns
  `;

  /**
   * Short alias for convenience
   */
  static readonly alias = 'alias';

  /**
   * Creates a new BubbleName instance.
   *
   * @param params - Operation parameters
   * @param context - Optional bubble execution context
   * @param instanceId - Unique instance identifier for debugging
   *
   * @example
   * ```typescript
   * const bubble = new BubbleName({
   *   operation: 'do_something',
   *   input: 'value'
   * });
   * ```
   */
  constructor(
    params: BubbleNameParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  /**
   * Tests if the provided credentials are valid.
   *
   * Makes a lightweight API call to verify authentication.
   * Uses minimal API quota and returns quickly.
   *
   * @returns Promise resolving to true if credentials are valid
   * @throws {AuthenticationError} When credentials are invalid
   *
   * @example
   * ```typescript
   * const isValid = await bubble.testCredential();
   * if (!isValid) {
   *   console.error('Invalid credentials');
   * }
   * ```
   */
  public async testCredential(): Promise<boolean> {
    // Implementation
  }

  /**
   * Selects the appropriate credential for this operation.
   *
   * @returns The credential string or undefined if not found
   * @throws {Error} When required credentials are missing
   *
   * @remarks
   * This method is called automatically during execution.
   * Override it to implement custom credential selection logic.
   */
  protected chooseCredential(): string | undefined {
    // Implementation
  }

  /**
   * Performs the main action of this bubble.
   *
   * This is the primary execution method that:
   * 1. Validates input parameters
   * 2. Authenticates with the service
   * 3. Executes the requested operation
   * 4. Processes and returns the result
   *
   * @param context - Optional bubble execution context
   * @returns Promise resolving to the operation result
   * @throws {ValidationError} When input parameters fail validation
   * @throws {AuthenticationError} When authentication fails
   * @throws {ApiError} When the service API returns an error
   * @throws {NetworkError} When network connectivity fails
   *
   * @remarks
   * Implements retry logic with exponential backoff for transient failures.
   * Maximum retries: 3 (configurable via params.maxRetries)
   *
   * @example
   * ```typescript
   * try {
   *   const result = await bubble.performAction();
   *   if (result.success) {
   *     console.log('Success:', result.data);
   *   }
   * } catch (error) {
   *   console.error('Failed:', error.message);
   * }
   * ```
   */
  protected async performAction(
    context?: BubbleContext
  ): Promise<BubbleNameResult> {
    // Implementation
  }

  /**
   * Helper method for common operations.
   *
   * @param input - The input value to process
   * @returns Processed output value
   * @throws {ProcessingError} When processing fails
   *
   * @example
   * ```typescript
   * const output = bubble.helperMethod('input');
   * ```
   */
  private helperMethod(input: string): string {
    // Implementation
  }
}
```

### Template 2: Tool Bubble

```typescript
/**
 * TOOL_NAME - Brief Description
 *
 * @module bubbles/tool-bubble/tool-name
 * @description
 * Description of what this tool does and when to use it.
 *
 * ## Features
 * - Feature 1
 * - Feature 2
 *
 * ## Input Format
 * Describe the expected input format.
 *
 * ## Output Format
 * Describe the output format.
 *
 * @example
 * Basic usage:
 * ```typescript
 * const tool = new ToolName({
 *   input: 'value'
 * });
 * const result = await tool.action();
 * ```
 */

import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';

/**
 * Tool input parameters.
 */
export type ToolNameParams = z.infer<typeof ToolNameParamsSchema>;

/**
 * Tool result type.
 */
export type ToolNameResult = z.infer<typeof ToolNameResultSchema>;

export class ToolNameTool extends ToolBubble<ToolNameParams, ToolNameResult> {
  static readonly type = 'tool' as const;
  static readonly bubbleName = 'tool-name';
  static readonly schema = ToolNameParamsSchema;
  static readonly resultSchema = ToolNameResultSchema;
  static readonly shortDescription = 'Brief description';
  static readonly longDescription = `Detailed description`;
  static readonly alias = 'tool';

  /**
   * Creates a new ToolName instance.
   *
   * @param params - Tool parameters
   * @param context - Optional execution context
   */
  constructor(params: ToolNameParams, context?: BubbleContext) {
    super(params, context);
  }

  /**
   * Executes the tool's main operation.
   *
   * @returns Promise resolving to the tool result
   * @throws {ValidationError} When input validation fails
   * @throws {ProcessingError} When processing fails
   *
   * @example
   * ```typescript
   * const result = await tool.performAction();
   * console.log(result.output);
   * ```
   */
  async performAction(): Promise<ToolNameResult> {
    // Implementation
  }
}
```

### Template 3: Workflow Bubble

```typescript
/**
 * WORKFLOW_NAME - Brief Description
 *
 * @module bubbles/workflow-bubble/workflow-name
 * @description
 * Description of this workflow and its purpose.
 *
 * ## Workflow Steps
 * 1. Step one description
 * 2. Step two description
 * 3. Step three description
 *
 * ## Preconditions
 * - Condition 1
 * - Condition 2
 *
 * ## Postconditions
 * - Result 1
 * - Result 2
 *
 * @example
 * Execute workflow:
 * ```typescript
 * const workflow = new WorkflowName({
 *   input: 'value',
 *   config: { option: 'value' }
 * });
 * const result = await workflow.execute();
 * ```
 */

import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';

/**
 * Workflow input parameters.
 */
export type WorkflowNameParams = z.infer<typeof WorkflowNameParamsSchema>;

/**
 * Workflow result type.
 */
export type WorkflowNameResult = z.infer<typeof WorkflowNameResultSchema>;

export class WorkflowNameWorkflow extends WorkflowBubble<WorkflowNameParams, WorkflowNameResult> {
  static readonly type = 'workflow' as const;
  static readonly bubbleName = 'workflow-name';
  static readonly schema = WorkflowNameParamsSchema;
  static readonly resultSchema = WorkflowNameResultSchema;
  static readonly shortDescription = 'Brief description';
  static readonly longDescription = `Detailed description`;

  /**
   * Executes the workflow.
   *
   * @returns Promise resolving to workflow result
   * @throws {WorkflowError} When workflow execution fails
   *
   * @remarks
   * This workflow orchestrates multiple bubbles in sequence.
   * Each step's output is passed to the next step.
   */
  async execute(): Promise<WorkflowNameResult> {
    // Implementation
  }
}
```

---

## JSDoc Tag Reference

### Standard Tags

#### @param
```typescript
/**
 * @param paramName - Description of the parameter
 * @param options - Configuration options (optional)
 * @param credentials - API credentials (injected at runtime)
 */
```

#### @returns
```typescript
/**
 * @returns Promise resolving to the result object with success flag and data
 */
```

#### @throws
```typescript
/**
 * @throws {ValidationError} When input validation fails
 * @throws {AuthenticationError} When credentials are invalid
 * @throws {ApiError} When the external API returns an error
 */
```

#### @example
```typescript
/**
 * @example
 * Basic usage:
 * ```typescript
 * const bubble = new MyBubble({ param: 'value' });
 * const result = await bubble.action();
 * ```
 *
 * @example
 * Advanced usage with options:
 * ```typescript
 * const bubble = new MyBubble({
 *   param: 'value',
 *   options: { retry: 3, timeout: 5000 }
 * });
 * ```
 */
```

#### @remarks
```typescript
/**
 * @remarks
 * Additional context about implementation details:
 * - Uses exponential backoff for retries
 * - Implements caching for performance
 * - Follows API rate limits automatically
 */
```

#### @see
```typescript
/**
 * @see {@link CredentialType} for available credential types
 * @see [External API Docs](https://api.example.com/docs)
 * @see RelatedClass for similar functionality
 */
```

#### @deprecated
```typescript
/**
 * @deprecated Use {@link NewMethod} instead
 * @removal 2.0.0
 *
 * This method will be removed in version 2.0.0.
 * Migration guide: https://docs.example.com/migration
 */
```

#### @since
```typescript
/**
 * @since 1.2.0
 *
 * Added in version 1.2.0 to support new feature.
 */
```

### TypeScript-Specific Tags

#### @template
```typescript
/**
 * @template T - The type parameter description
 * @template U - Second type parameter
 */
function genericFunction<T, U>(param1: T, param2: U): Promise<T & U> {
  // Implementation
}
```

#### @type
```typescript
/**
 * @type {string | number}
 */
const unionValue = 'value';

/**
 * @type {Record<string, unknown>}
 */
const dataObject = {};
```

---

## Inline Comment Guidelines

### When to Add Inline Comments

#### 1. Complex Algorithms
```typescript
// Calculate fibonacci sequence using dynamic programming
// Time complexity: O(n), Space complexity: O(n)
// Uses memoization to avoid redundant calculations
const fib = (n: number): number => {
  if (n <= 1) return n;
  const dp = [0, 1];

  // Build sequence iteratively
  for (let i = 2; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2]; // Each number is sum of previous two
  }

  return dp[n];
};
```

#### 2. Regular Expressions
```typescript
// Match email addresses with following format:
// local-part@domain.tld
// - local-part: alphanumeric, dots, hyphens, underscores
// - domain: alphanumeric parts separated by dots
// - tld: 2-6 alphabetic characters
const emailRegex = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$/;
```

#### 3. Security Decisions
```typescript
// SECURITY: Validate URL to prevent SSRF attacks
// Block internal IP ranges and private networks
// This prevents attackers from scanning internal infrastructure
if (isInternalUrl(url)) {
  throw new Error('Internal URLs are not allowed');
}
```

#### 4. Non-Obvious Code
```typescript
// Use bitwise XOR to swap values without temporary variable
// a = a ^ b, b = a ^ b, a = a ^ b
let a = 5, b = 10;
a = a ^ b; // a now contains 5 ^ 10
b = a ^ b; // b becomes original a (5)
a = a ^ b; // a becomes original b (10)
```

#### 5. Performance Optimizations
```typescript
// Pre-allocate array capacity for better performance
// Avoids dynamic resizing during push operations
const results: string[] = new Array(expectedSize);

// Use Set for O(1) lookups instead of array's O(n)
const uniqueItems = new Set<string>();
```

### Comment Style

#### Do's
```typescript
// ✅ GOOD: Clear, explanatory
// Normalize hostname to lowercase for case-insensitive comparison
const hostname = url.hostname.toLowerCase();

// ✅ GOOD: Explains why
// Use Set instead of array for O(1) lookup performance
const cache = new Set<string>();

// ✅ GOOD: Warns about consequences
// WARNING: This operation cannot be undone. Data will be permanently deleted.
await deleteAllRecords();
```

#### Don'ts
```typescript
// ❌ BAD: Restates the obvious
// Increment counter
counter++;

// ❌ BAD: Too vague
// Handle result
handleResult(result);

// ❌ BAD: No explanation
// Complex logic here
const result = data.map(x => x * 2).filter(x => x > 10).reduce((a, b) => a + b, 0);
```

---

## Type Documentation

### Interface Documentation

```typescript
/**
 * Configuration options for API requests.
 *
 * @property timeout - Request timeout in milliseconds (default: 30000)
 * @property retries - Maximum number of retry attempts (default: 3)
 * @property retryDelay - Base delay between retries in milliseconds (default: 1000)
 * @property validateStatus - Custom status code validator (default: 2xx only)
 *
 * @remarks
 * Retry logic uses exponential backoff:
 * delay = retryDelay * 2^(attempt - 1) + random jitter
 *
 * @example
 * ```typescript
 * const config: RequestConfig = {
 *   timeout: 60000,
 *   retries: 5,
 *   retryDelay: 2000,
 *   validateStatus: (status) => status >= 200 && status < 400
 * };
 * ```
 */
export interface RequestConfig {
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  validateStatus?: (status: number) => boolean;
}
```

### Discriminated Union Documentation

```typescript
/**
 * Parameters for different message operations.
 *
 * This discriminated union uses the `operation` field to determine
 * the specific schema for each operation type.
 *
 * @example
 * Send message:
 * ```typescript
 * const params: MessageParams = {
 *   operation: 'send',
 *   recipient: 'user123',
 *   content: 'Hello!'
 * };
 * ```
 *
 * @example
 * List messages:
 * ```typescript
 * const params: MessageParams = {
 *   operation: 'list',
 *   limit: 50,
 *   offset: 0
 * };
 * ```
 */
export type MessageParams =
  | {
      operation: 'send';
      recipient: string;
      content: string;
      options?: MessageOptions;
    }
  | {
      operation: 'list';
      limit?: number;
      offset?: number;
      filter?: MessageFilter;
    }
  | {
      operation: 'delete';
      messageId: string;
    };
```

### Generic Type Documentation

```typescript
/**
 * A cache that stores values with automatic expiration.
 *
 * @template K - The type of cache keys (must extend string | number)
 * @template V - The type of cached values
 *
 * @example
 * ```typescript
 * const cache = new ExpiringCache<string, User>({
 *   ttl: 3600000, // 1 hour
 *   maxSize: 1000
 * });
 *
 * cache.set('user123', userObject);
 * const user = cache.get('user123');
 * ```
 */
export class ExpiringCache<K extends string | number, V> {
  // Implementation
}
```

---

## Security Documentation

### Security-Focused Comment Template

```typescript
/**
 * Validates and sanitizes user input to prevent security vulnerabilities.
 *
 * ## Security Considerations
 * - **XSS Prevention**: All HTML tags are stripped from user input
 * - **SQL Injection**: Parameterized queries are used (no string concatenation)
 * - **Path Traversal**: File paths are validated and normalized
 * - **SSRF Protection**: Internal IP ranges are blocked
 *
 * @param input - Raw user input
 * @returns Sanitized and validated output
 * @throws {ValidationError} When input contains malicious patterns
 *
 * @remarks
 * This function implements defense in depth:
 * 1. Input validation against whitelist
 * 2. Sanitization of known dangerous patterns
 * 3. Length limits to prevent DoS
 * 4. Character encoding validation
 */
function sanitizeInput(input: string): string {
  // SECURITY: Validate against XSS attacks
  // Remove all HTML tags and special characters
  const sanitized = input.replace(/<[^>]*>/g, '');

  // SECURITY: Prevent SQL injection
  // Only allow alphanumeric and safe characters
  if (!/^[a-zA-Z0-9\s\-_.]+$/.test(sanitized)) {
    throw new ValidationError('Input contains invalid characters');
  }

  // SECURITY: Limit length to prevent DoS
  const MAX_LENGTH = 1000;
  if (sanitized.length > MAX_LENGTH) {
    throw new ValidationError(`Input exceeds maximum length of ${MAX_LENGTH}`);
  }

  return sanitized;
}
```

---

## Example Documentation

### Multi-Example Template

```typescript
/**
 * Performs advanced data transformation.
 *
 * @example
 * Basic transformation:
 * ```typescript
 * const result = await transform({
 *   input: [1, 2, 3],
 *   operation: 'double'
 * });
 * // Result: [2, 4, 6]
 * ```
 *
 * @example
 * With custom options:
 * ```typescript
 * const result = await transform({
 *   input: [1, 2, 3],
 *   operation: 'map',
 *   options: {
 *     mapper: (x) => x * x,
 *     parallel: true
 *   }
 * });
 * // Result: [1, 4, 9]
 * ```
 *
 * @example
 * Error handling:
 * ```typescript
 * try {
 *   const result = await transform({ input: data, operation: 'sort' });
 * } catch (error) {
 *   if (error instanceof ValidationError) {
 *     console.error('Invalid input:', error.message);
 *   } else if (error instanceof ProcessingError) {
 *     console.error('Processing failed:', error.message);
 *   }
 * }
 * ```
 */
```

---

## Best Practices Summary

1. **Document Why, Not Just What**
   - Explain decisions and reasoning
   - Document trade-offs
   - Note edge cases

2. **Keep Examples Simple Yet Complete**
   - Show import statements
   - Include error handling
   - Use realistic values

3. **Use Consistent Formatting**
   - Follow template structure
   - Use same comment style throughout
   - Maintain consistent tag order

4. **Update Documentation with Code**
   - Keep docs in sync with implementation
   - Document breaking changes
   - Mark deprecated features

5. **Focus on User Value**
   - Answer common questions
   - Anticipate confusion points
   - Provide practical guidance

---

## Quick Reference

### Essential Tags for Every Method
- `@param` - For each parameter
- `@returns` - For return value
- `@throws` - For error conditions
- `@example` - For complex operations

### Essential Comments for Every File
- File-level JSDoc with description
- Module purpose and features
- Usage examples
- Configuration options

### When to Add Examples
- Public API methods
- Complex operations
- Non-obvious usage patterns
- Error scenarios

---

**End of Documentation Templates**

For questions or suggestions, please open an issue or submit a PR.

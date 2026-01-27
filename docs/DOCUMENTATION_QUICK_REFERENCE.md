# BubbleLab Documentation Quick Reference

**Version:** 1.0.0
**Last Updated:** 2025-01-18

A quick reference guide for documenting BubbleLab bubbles to professional standards.

## TL;DR - The Essentials

### Every Public Method Needs
```typescript
/**
 * Brief one-line description.
 *
 * Detailed description of what the method does and when to use it.
 *
 * @param paramName - Description of parameter
 * @returns Description of return value
 * @throws {ErrorType} When error condition occurs
 *
 * @example
 * ```typescript
 * const result = await method({ param: 'value' });
 * ```
 */
```

### Every Complex Type Needs
```typescript
/**
 * Type name with purpose.
 *
 * @property prop1 - Description
 * @property prop2 - Description
 *
 * @remarks
 * Important notes about usage.
 *
 * @example
 * ```typescript
 * const data: TypeName = { prop1: 'value1', prop2: 'value2' };
 * ```
 */
```

### Every Security Check Needs
```typescript
// SECURITY: Brief explanation of threat and mitigation
// Rationale: Why this security measure is necessary
```

---

## Checklist

### File-Level (Do Once Per File)
- [ ] File header JSDoc with module description
- [ ] Feature list
- [ ] Use cases (3-5 examples)
- [ ] Configuration requirements
- [ ] External documentation links
- [ ] Version and author info

### Method-Level (Per Public Method)
- [ ] JSDoc comment block
- [ ] @param for each parameter with description
- [ ] @returns with type and description
- [ ] @throws for error conditions
- [ ] @example for complex operations
- [ ] @remarks for implementation details

### Type-Level (Per Complex Type)
- [ ] Interface/type description
- [ ] @property for each field
- [ ] Type usage examples
- [ ] @remarks for important notes

### Inline Comments (Where Needed)
- [ ] Complex algorithms explained
- [ ] Security decisions documented
- [ ] Regex patterns broken down
- [ ] Performance notes added
- [ ] Non-obvious code clarified

---

## Common Patterns

### Pattern 1: Async Method
```typescript
/**
 * Performs async operation with retry logic.
 *
 * @param input - The input data to process
 * @returns Promise resolving to processed result
 * @throws {ValidationError} When input validation fails
 * @throws {ApiError} When external API fails after retries
 *
 * @example
 * ```typescript
 * const result = await performAction({ data: 'value' });
 * console.log(result.processedData);
 * ```
 *
 * @remarks
 * Implements exponential backoff:
 * - Base delay: 1s
 * - Max delay: 32s
 * - Max retries: 3 (configurable)
 */
async performAction(input: InputType): Promise<ResultType> {
  // Implementation
}
```

### Pattern 2: Discriminated Union
```typescript
/**
 * Parameters for different operations.
 *
 * Uses `operation` field to determine schema.
 *
 * @example
 * Operation A:
 * ```typescript
 * const params: Params = { operation: 'opA', value: 'x' };
 * ```
 *
 * @example
 * Operation B:
 * ```typescript
 * const params: Params = { operation: 'opB', items: [1, 2, 3] };
 * ```
 */
export type Params =
  | { operation: 'opA'; value: string }
  | { operation: 'opB'; items: number[] };
```

### Pattern 3: Class Documentation
```typescript
/**
 * BubbleName - Brief description.
 *
 * @module bubbles/service-bubble/bubble-name
 * @description
 * Detailed description of functionality.
 *
 * ## Features
 * - Feature 1
 * - Feature 2
 *
 * ## Authentication
 * Requires CREDENTIAL_TYPE.
 *
 * ## Example
 * ```typescript
 * const bubble = new BubbleName({
 *   param: 'value',
 *   credentials: { [CredentialType.TYPE]: 'key' }
 * });
 * ```
 */
export class BubbleName extends ServiceBubble<Params, Result> {
  // Implementation
}
```

---

## Security Documentation Templates

### SSRF Protection
```typescript
// SECURITY: Block internal IP ranges to prevent SSRF attacks
// Threat: Attacker probes internal network or accesses cloud metadata
// Mitigation: Reject private IPs (RFC 1918), localhost, link-local
// Impact: Prevents unauthorized network reconnaissance

// RFC 1918 private ranges:
// - 10.0.0.0/8     (10.0.0.0 - 10.255.255.255)
// - 172.16.0.0/12  (172.16.0.0 - 172.31.255.255)
// - 192.168.0.0/16 (192.168.0.0 - 192.168.255.255)
const privateIpPatterns = [/^10\./, /^172\.(1[6-9]|2\d|3[01])\./, /^192\.168\./];
```

### Input Validation
```typescript
// SECURITY: Validate and sanitize user input
// Threat: XSS via script injection, SQL injection
// Mitigation: Whitelist validation, length limits, sanitization
// Impact: Prevents code execution and data breaches

if (!/^[a-zA-Z0-9\s\-_.]+$/.test(input)) {
  throw new ValidationError('Input contains invalid characters');
}
```

### Path Traversal Prevention
```typescript
// SECURITY: Prevent path traversal attacks
// Threat: Access arbitrary files on filesystem
// Mitigation: Block "..", absolute paths, validate with normalize
// Impact: Prevents unauthorized file access

if (filePath.includes('..') || path.isAbsolute(filePath)) {
  throw new ValidationError('Invalid file path');
}
```

### SQL Injection Prevention
```typescript
// SECURITY: Use parameterized queries to prevent SQL injection
// Threat: Inject malicious SQL via user input
// Mitigation: Always use placeholders, never concatenate strings
// Impact: Prevents database compromise

// ❌ BAD: String concatenation
// db.query(`SELECT * FROM users WHERE id = '${userId}'`);

// ✅ GOOD: Parameterized
// db.query('SELECT * FROM users WHERE id = $1', [userId]);
```

---

## Comment Style Guide

### ✅ DO: Explain Why
```typescript
// Use Set for O(1) lookups instead of array's O(n)
const cache = new Set<string>();
```

### ✅ DO: Warn About Consequences
```typescript
// WARNING: This operation cannot be undone
// Data will be permanently deleted
await deleteAllRecords();
```

### ✅ DO: Document Security
```typescript
// SECURITY: Validate URL to prevent SSRF
if (isInternalUrl(url)) {
  throw new Error('Internal URLs blocked');
}
```

### ❌ DON'T: Restate the Obvious
```typescript
// Bad: Increment counter
counter++;

// Good: Increment to track retry attempts
retryCount++;
```

### ❌ DON'T: Be Vague
```typescript
// Bad: Handle result
handleResult(result);

// Good: Parse API response and extract data
const parsed = parseApiResponse(result);
```

---

## JSDoc Tags Quick Reference

### Basic Tags
```typescript
/**
 * @param name - Description
 * @returns Description of return value
 * @throws {ErrorType} When error occurs
 * @example Description and code
 * @remarks Additional context
 * @see {@link OtherType} or URL
 */
```

### TypeScript Tags
```typescript
/**
 * @template T - Type parameter description
 * @type {string | number}
 */
```

### Metadata Tags
```typescript
/**
 * @deprecated Since 1.0.0, use NewMethod instead
 * @since 1.0.0
 * @version 1.0.0
 * @author Author Name
 */
```

---

## Example Quality Levels

### Level 1: Minimal (Acceptable for Simple Methods)
```typescript
/**
 * Gets the user by ID.
 *
 * @param userId - The user ID
 * @returns The user object or null
 */
function getUser(userId: string): User | null {
  // Implementation
}
```

### Level 2: Good (Standard for Most Methods)
```typescript
/**
 * Gets the user by ID from the database.
 *
 * @param userId - The unique user identifier
 * @returns The user object if found, null otherwise
 * @throws {DatabaseError} When database query fails
 *
 * @example
 * ```typescript
 * const user = getUser('user123');
 * if (user) {
 *   console.log(user.name);
 * }
 * ```
 */
function getUser(userId: string): Promise<User | null> {
  // Implementation
}
```

### Level 3: Excellent (Required for Complex Methods)
```typescript
/**
 * Gets the user by ID with caching and fallback.
 *
 * Implements a multi-layer caching strategy:
 * 1. Check memory cache (fastest)
 * 2. Check Redis cache (fast)
 * 3. Query database (slow)
 *
 * @param userId - The unique user identifier (UUID format)
 * @param options - Configuration options
 * @param options.useCache - Enable caching (default: true)
 * @param options.ttl - Cache TTL in seconds (default: 3600)
 * @returns Promise resolving to user object or null if not found
 * @throws {ValidationError} When userId format is invalid
 * @throws {DatabaseError} When database query fails
 * @throws {CacheError} When cache operations fail
 *
 * @example
 * Basic usage:
 * ```typescript
 * const user = await getUser('123e4567-e89b-12d3-a456-426614174000');
 * ```
 *
 * @example
 * With options:
 * ```typescript
 * const user = await getUser('user123', {
 *   useCache: false,  // Bypass cache
 *   ttl: 7200        // Custom TTL
 * });
 * ```
 *
 * @remarks
 * **Cache Invalidation:**
 * - Cache is invalidated on user updates
 * - Use `useCache: false` to force fresh data
 *
 * **Performance:**
 * - Memory cache: ~1ms
 * - Redis cache: ~10ms
 * - Database: ~100ms
 */
async getUser(
  userId: string,
  options: { useCache?: boolean; ttl?: number } = {}
): Promise<User | null> {
  // Implementation
}
```

---

## Common Mistakes to Avoid

### 1. Missing @param Descriptions
❌ **Bad:**
```typescript
/**
 * @param userId
 * @param options
 */
```

✅ **Good:**
```typescript
/**
 * @param userId - The unique user identifier
 * @param options - Configuration options for the query
 */
```

### 2. Vague @throws
❌ **Bad:**
```typescript
/**
 * @throws {Error}
 */
```

✅ **Good:**
```typescript
/**
 * @throws {ValidationError} When input parameters fail validation
 * @throws {AuthenticationError} When credentials are invalid
 */
```

### 3. No Examples for Complex APIs
❌ **Bad:**
```typescript
/**
 * Sends a message to Slack.
 */
```

✅ **Good:**
```typescript
/**
 * Sends a message to Slack.
 *
 * @example
 * Basic message:
 * ```typescript
 * await slack.sendMessage({ channel: 'general', text: 'Hello' });
 * ```
 *
 * @example
 * With blocks:
 * ```typescript
 * await slack.sendMessage({
 *   channel: 'general',
 *   blocks: [{ type: 'section', text: { type: 'plain_text', text: 'Hello' } }]
 * });
 * ```
 */
```

### 4. Missing Type Documentation
❌ **Bad:**
```typescript
interface RequestOptions {
  timeout?: number;
  retries?: number;
}
```

✅ **Good:**
```typescript
/**
 * Configuration options for API requests.
 *
 * @property timeout - Request timeout in milliseconds (default: 30000)
 * @property retries - Maximum number of retry attempts (default: 3)
 *
 * @remarks
 * Retry logic uses exponential backoff.
 */
interface RequestOptions {
  timeout?: number;
  retries?: number;
}
```

### 5. Uncommented Security Code
❌ **Bad:**
```typescript
if (url.includes('localhost')) {
  return false;
}
```

✅ **Good:**
```typescript
// SECURITY: Block localhost to prevent SSRF attacks
// Prevents access to internal services
if (url.includes('localhost')) {
  return false;
}
```

---

## Documentation Workflow

### Step 1: Analyze the Code
1. Read through the entire file
2. Identify public methods
3. Note complex algorithms
4. Find security-critical code
5. List types that need documentation

### Step 2: Add File-Level Docs
1. Create file header JSDoc
2. List features and use cases
3. Document configuration
4. Add external links
5. Include overview example

### Step 3: Document Methods
1. Add JSDoc to each public method
2. Document all parameters
3. Document return type
4. List error conditions
5. Add examples for complex methods

### Step 4: Document Types
1. Add interface descriptions
2. Document each property
3. Add usage examples
4. Note any constraints
5. Explain discriminated unions

### Step 5: Add Inline Comments
1. Explain complex algorithms
2. Document security decisions
3. Break down regex patterns
4. Add performance notes
5. Clarify non-obvious code

### Step 6: Review
1. Check all @param tags
2. Verify @returns tags
3. Test all examples
4. Ensure consistency
5. Run linter

---

## Quick Reference Card

### Essential Tags (Memorize These)
- `@param name - desc` - Parameter description
- `@returns desc` - Return value description
- `@throws {Type} desc` - Error condition
- `@example desc` - Usage example
- `@remarks desc` - Additional context

### Comment Types (When to Use)
- File header: Every file
- Method JSDoc: All public methods
- Type docs: All complex types
- Inline: Complex/security-critical code
- TODO: Future improvements
- FIXME: Known issues
- HACK: Temporary solutions

### Security Flags (ALWAYS Document)
- SSRF protection
- Input validation
- Path traversal prevention
- SQL injection prevention
- XSS prevention
- Authentication checks
- Authorization checks

---

## Resources

### Internal Documentation
- [Full Templates](./DOCUMENTATION_TEMPLATES.md)
- [Improvement Report](./WAVE2C_DOCUMENTATION_REPORT.md)
- [Tracking Report](./DOCUMENTATION_IMPROVEMENT_WAVE2C.md)
- [Bubbles README](../BubbleLab/packages/bubble-core/src/bubbles/README.md)

### External Resources
- [JSDoc Official](https://jsdoc.app/)
- [TypeScript JSDoc](https://www.typescriptlang.org/docs/handbook/jsdoc-supported-types.html)
- [OWASP Security](https://owasp.org/)
- [Google Style Guide](https://google.github.io/styleguide/tsguide.html)

---

## Tips for Efficiency

### 1. Use Snippets
Create VS Code snippets for common patterns:
```json
{
  "Method JSDoc": {
    "prefix": "jsdoc-method",
    "body": [
      "/**",
      " * ${1:brief description}.",
      " *",
      " * @param ${2:param} - ${3:description}",
      " * @returns ${4:description}",
      " * @throws ${5:ErrorType} When ${6:condition}",
      " *",
      " * @example",
      " * ```typescript",
      " * const result = await ${7:method}(${8:args});",
      " * ```",
      " */"
    ]
  }
}
```

### 2. Batch Similar Methods
Document all similar methods at once:
- Copy base template
- Adjust for each method
- Ensures consistency
- Saves time

### 3. Generate First, Enhance Second
1. Generate basic JSDoc automatically
2. Add descriptions and examples manually
3. Focus on high-value improvements

### 4. Review in Iterations
1. First pass: Basic JSDoc on all methods
2. Second pass: Add examples and @throws
3. Third pass: Add inline comments
4. Final pass: Polish and review

---

## Final Checklist

Before marking documentation as complete:

### Coverage
- [ ] All public methods have JSDoc
- [ ] All @param tags have descriptions
- [ ] All @returns tags are complete
- [ ] Error conditions documented
- [ ] Complex types explained

### Quality
- [ ] Examples are accurate and runnable
- [ ] Security decisions documented
- [ ] Inline comments clarify complex code
- [ ] No typos or grammatical errors
- [ ] Consistent style throughout

### Usability
- [ ] File-level overview is clear
- [ ] Use cases are practical
- [ ] Configuration is documented
- [ ] External links work
- [ ] Examples cover common scenarios

---

**Remember:** Good documentation is an investment in developer productivity and code maintainability. Take the time to do it right!

**Need Help?** Refer to the full templates or ask the team.

---

**Last Updated:** 2025-01-18
**Version:** 1.0.0

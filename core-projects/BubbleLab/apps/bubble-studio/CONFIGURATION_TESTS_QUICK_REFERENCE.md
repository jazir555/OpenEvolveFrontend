# Configuration Tests - Quick Reference

## Test Suite Overview

**Total Tests:** 170 tests across 10 files
**Success Rate:** 100% (170/170 passed)
**Execution Time:** ~11 seconds

---

## Test Files Summary

| File | Tests | Description |
|------|-------|-------------|
| `environment.test.ts` | 31 | Environment variable configuration and validation |
| `health-validation.test.ts` | 22 | Service health check validation |
| `url-resolution.test.ts` | 25 | URL resolution and normalization |
| `integration.test.ts` | 16 | Integration and startup tests |
| Other existing tests | 76 | Pre-existing test suites |

---

## Running the Tests

```bash
# Run all tests
npm test -- --run

# Run specific test file
npm test -- environment.test.ts --run

# Run with coverage
npm test -- --coverage --run

# Run in watch mode
npm test
```

---

## What the Tests Validate

### 1. Production Mode Safety
✅ Application fails without required config
✅ Clear error messages
✅ No silent fallbacks

### 2. Development Mode Flexibility
✅ Warns about missing config
✅ Falls back to localhost:8000
✅ Application starts successfully

### 3. URL Validation
✅ Accepts HTTP/HTTPS URLs
✅ Rejects invalid formats
✅ Includes invalid URL in error

### 4. URL Normalization
✅ Removes trailing slashes
✅ Handles multiple slashes
✅ Preserves paths and query params

### 5. Environment Variable Priority
```
VITE_EVOLUTION_API_URL (highest)
    ↓ (if not set)
VITE_GATEWAY_URL (fallback)
    ↓ (if not set)
localhost:8000 (dev only)
```

### 6. Health Validation
✅ Identifies healthy/unhealthy services
✅ Lists all unhealthy services
✅ Supports skip validation
✅ Handles timeouts and retries

---

## Test Coverage by Bug

### Bug #1: Environment Variable Validation
**Status:** ✅ FIXED AND VALIDATED

**Tests:**
- `should require VITE_EVOLUTION_API_URL in production`
- `should provide clear production error message`
- `should NOT fallback to localhost in production`
- `should warn in development without config`
- `should fallback to localhost:8000 in development`

**File:** `environment.test.ts`

---

### Bug #6: Service Health Validation
**Status:** ✅ FIXED AND VALIDATED

**Tests:**
- `should identify healthy services`
- `should identify unhealthy services`
- `should pass validation when all services are healthy`
- `should throw error if any service is unhealthy`
- `should list all unhealthy services in error message`
- `should bypass validation when skipValidation=true`
- `should handle timeout errors gracefully`

**File:** `health-validation.test.ts`

---

### Bug #8: Configuration Error Handling
**Status:** ✅ FIXED AND VALIDATED

**Tests:**
- `should reject invalid URL format`
- `should reject URL without protocol`
- `should include invalid URL in error message`
- `should provide actionable error messages`
- `should include troubleshooting information in errors`

**File:** `url-resolution.test.ts`, `integration.test.ts`

---

## Key Test Scenarios

### Production Startup Without Config
```typescript
// This should FAIL
MODE=production
VITE_EVOLUTION_API_URL=""
// Result: Error - "CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set"
```

### Development Startup Without Config
```typescript
// This should WARN but SUCCEED
MODE=development
VITE_EVOLUTION_API_URL=""
// Result: Warning + fallback to http://localhost:8000
```

### Invalid URL Format
```typescript
// This should FAIL
VITE_EVOLUTION_API_URL="not-a-valid-url"
// Result: Error - "Invalid VITE_EVOLUTION_API_URL format: not-a-valid-url"
```

### Healthy Service Check
```typescript
// All services healthy
const results = [
  { service: 'evolution-api', healthy: true, latency: 50 },
  { service: 'gateway', healthy: true, latency: 75 }
];
// Result: Validation passes
```

### Unhealthy Service Check
```typescript
// One service unhealthy
const results = [
  { service: 'evolution-api', healthy: true, latency: 50 },
  { service: 'gateway', healthy: false, error: 'Connection refused' }
];
// Result: Error - lists 'gateway: Connection refused'
```

---

## Test Organization

```
src/tests/configuration/
├── environment.test.ts       (31 tests) - Env var validation
├── health-validation.test.ts (22 tests) - Health checks
├── url-resolution.test.ts    (25 tests) - URL handling
└── integration.test.ts       (16 tests) - Integration tests
```

---

## Common Test Patterns

### Testing Error Conditions
```typescript
it('should throw error for invalid config', () => {
  const env = { MODE: 'production', VITE_EVOLUTION_API_URL: '' };

  expect(() => {
    // code that should throw
  }).toThrow();

  // OR

  try {
    // code that should throw
    expect.fail('Should have thrown');
  } catch (error: any) {
    expect(error.message).toContain('expected text');
  }
});
```

### Testing Console Output
```typescript
it('should warn in development', () => {
  const consoleWarnSpy = vi.spyOn(console, 'warn');

  // code that should warn

  expect(consoleWarnSpy).toHaveBeenCalledWith(
    expect.stringContaining('expected text')
  );

  consoleWarnSpy.mockRestore();
});
```

---

## Maintenance Tips

1. **Adding New Tests:**
   - Follow existing patterns
   - Test both success and failure cases
   - Include edge cases
   - Verify error messages

2. **Test Naming:**
   - Use clear, descriptive names
   - Format: `should <expected behavior> when <condition>`
   - Example: `should throw error when URL is invalid`

3. **Error Messages:**
   - Always test error message content
   - Verify messages are actionable
   - Check that invalid values are included

4. **Environment Variables:**
   - Test with empty string
   - Test with whitespace
   - Test with invalid values
   - Test with valid values

---

## Success Criteria

✅ All 170 tests pass
✅ 100% test success rate
✅ Configuration validation works correctly
✅ Production mode is protected
✅ Development mode is flexible
✅ Error messages are clear and actionable
✅ URL validation is robust
✅ Health checks work correctly

---

## Next Steps

The configuration validation system is fully tested and working correctly. All configuration bugs have been addressed:

1. ✅ Bug #1: Environment variable validation
2. ✅ Bug #6: Service health validation
3. ✅ Bug #8: Configuration error handling

**Status:** Ready for production deployment

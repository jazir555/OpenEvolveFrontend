# Bubble Refactoring - Quick Start Guide

## For the Next Developer

You're continuing the systematic refactoring of 117+ bubbles. Here's what you need to know:

## Current Status

✅ **Completed:**
- Documentation created (3 comprehensive guides)
- Common utilities verified (7 modules, 1,678 lines)
- slack.ts partially refactored (file path validation, error handling, JSDoc)
- http.ts partially refactored (imports added, helper function created)

⏳ **Next:**
- Complete slack.ts and http.ts refactoring
- Refactor postgresql.ts, ai-agent.ts, airtable.ts
- Then batch-process remaining 113 files

## Quick Reference

### Common Utilities Location
```
BubbleLab/packages/bubble-core/src/bubbles/common/
```

### Key Imports to Add
```typescript
import {
  validateEmail,
  validateUrl,
  validateFilePath,
  validateNonEmptyString,
  ValidationError as CommonValidationError
} from '../common/validators.js';

import {
  AuthenticationError,
  ExternalServiceError,
  ValidationError,
  NetworkError,
  TimeoutError
} from '../common/error-handlers.js';

import {
  retryWithBackoff,
  withTimeout,
  CircuitBreaker
} from '../common/retry.js';
```

## 5 Essential Refactoring Patterns

### Pattern 1: File Path Validation
**Before:** 46 lines of inline validation
**After:**
```typescript
import { validateFilePath, ValidationError as CommonValidationError } from '../common/validators.js';

try {
  validateFilePath(file_path, false); // false = no absolute paths
} catch (error) {
  if (error instanceof CommonValidationError) {
    return { ok: false, error: error.message, success: false };
  }
  throw error;
}
```
**Savings:** ~42 lines

### Pattern 2: Error Handling
**Before:**
```typescript
throw new Error('Authentication failed');
```

**After:**
```typescript
import { AuthenticationError } from '../common/error-handlers.js';
throw new AuthenticationError('Authentication failed');
```

### Pattern 3: URL Validation
**Before:** 80+ lines of SSRF prevention logic
**After:** Use custom helper (see http.ts validateUrlSsrf function)

### Pattern 4: Add JSDoc
```typescript
/**
 * Test the validity of the credential
 * @returns Promise that resolves to true if credential is valid
 * @throws AuthenticationError if credentials are invalid
 */
public async testCredential(): Promise<boolean> {
  // implementation
}
```

### Pattern 5: Retry Logic
**Before:** 10+ lines of retry code
**After:**
```typescript
import { retryWithBackoff } from '../common/retry.js';

return await retryWithBackoff(
  () => makeRequest(),
  { maxAttempts: 3, baseDelayMs: 1000, operation: 'API Call' }
);
```
**Savings:** ~10 lines

## Priority Order

### Do These First (High Impact):
1. ✅ slack.ts (partially done)
2. ⏳ http.ts (partially done - needs schema update)
3. ⏳ postgresql.ts
4. ⏳ ai-agent.ts
5. ⏳ airtable.ts

### Then Batch Process:
6. All 30 Apify actors (similar patterns)
7. All 40 service bubbles
8. All 30 tool bubbles
9. All 21 workflow templates

## Time Estimates

- Critical bubbles: 3-4 hours
- Apify actors: 4-5 hours (batch processing)
- Service bubbles: 4-5 hours
- Tool bubbles: 2-3 hours
- Workflows: 1-2 hours

**Total remaining:** ~14-19 hours

## Realistic Targets

**Original estimate:** 14,200 lines (11% reduction)
**More realistic:** 6,000-8,000 lines (4-6% reduction)

Quality improvements are more valuable than raw line count:
- Better error types
- Consistent validation
- Improved maintainability
- Better developer experience

## Testing Checklist

After refactoring each file:
1. ✅ Run TypeScript compiler: `npm run type-check`
2. ✅ Run existing tests: `npm test -- <file>.test.ts`
3. ✅ Manual smoke test if critical bubble
4. ✅ Check for console errors
5. ✅ Verify error messages still make sense

## Common Pitfalls to Avoid

1. ❌ Don't forget to import common utilities
2. ❌ Don't change the logic, just the implementation
3. ❌ Don't skip JSDoc comments
4. ❌ Don't forget to test after each change
5. ❌ Don't refactor too many files at once without testing

## Helpful Commands

```bash
# Count lines in a file
wc -l packages/bubble-core/src/bubbles/service-bubble/slack.ts

# Find all bubble files
find packages/bubble-core/src/bubbles -name "*.ts" ! -name "*.test.ts"

# Run TypeScript check
npm run type-check

# Run tests
npm test

# Run specific test
npm test -- slack.test.ts
```

## Documentation Files

Read these for detailed context:
1. **P3_REFACTORING_GUIDE.md** - Comprehensive patterns and examples
2. **REFACTORING_PROGRESS_REPORT.md** - Detailed progress tracking
3. **BUBBLE_REFACTORING_FINAL_SUMMARY.md** - Complete analysis

## Pro Tips

1. **Start with similar files** - Once you figure out one Apify actor, the rest are similar
2. **Batch process** - Do all files of one type in one session
3. **Test frequently** - Don't wait until you've done 10 files to test
4. **Copy patterns** - Use the refactored slack.ts as a template
5. **Track progress** - Update the todo list as you go

## Quick Win: Complete These Now

If you have 1 hour:
- Complete slack.ts refactoring (add JSDoc to remaining methods)
- Complete http.ts refactoring (update schema to use validateUrlSsrf)

If you have 3 hours:
- Above + refactor postgresql.ts and ai-agent.ts

If you have a full day:
- Above + batch refactor all 30 Apify actors

## Need Help?

1. Check the documentation files
2. Look at refactored slack.ts as an example
3. Refer to common utilities source code
4. Run tests to verify your changes

## Good Luck! 🚀

The hard work (documentation, patterns, utilities) is done. You just need to apply the patterns to the remaining files.

Focus on:
- Consistency over perfection
- Quality over speed
- Testing over guessing

You've got this!

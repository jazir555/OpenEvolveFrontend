# Quick Reference: Critical Bug Fixes

## What Changed

Three critical configuration bugs were fixed to comply with the Federation Constitution.

### 1. KnowledgeEngineBubble - Dynamic API URL
**Before**: Hardcoded `http://localhost:8000`
**After**: Resolves from `OPENEVOLVE_API_URL` environment variable

### 2. env.ts - Production Validation
**Before**: Silent fallback to localhost
**After**: Throws error in production if `VITE_EVOLUTION_API_URL` is missing

### 3. createOpenEvolveIntegration - Health Validation
**Before**: Sync function, no validation
**After**: Async function with automatic health checks at startup

---

## Required Environment Variables

### Production (Required - App Will Crash If Missing)
```bash
# Server-side (Node.js/Docker)
OPENEVOLVE_API_URL=https://api.openevolve.com

# Client-side (Vite build)
VITE_EVOLUTION_API_URL=https://api.openevolve.com
# OR
VITE_GATEWAY_URL=https://api.openevolve.com
```

### Development (Optional - Falls Back to Localhost)
```bash
# If not set, will use http://localhost:8000 with a warning
```

---

## Code Changes Required

### Update createOpenEvolveIntegration Calls

**Before**:
```typescript
const integration = createOpenEvolveIntegration({
  knowledgeBackend: 'qdrant'
});
```

**After**:
```typescript
const integration = await createOpenEvolveIntegration({
  knowledgeBackend: 'qdrant'
});
```

### Unit Tests (Skip Validation)
```typescript
const integration = await createOpenEvolveIntegration(
  { knowledgeBackend: 'qdrant' },
  true // skipValidation for testing
);
```

---

## Error Messages

### Missing Config in Production
```
CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set.
This is a required configuration for production.

Please set one of the following:
  - VITE_EVOLUTION_API_URL (preferred)
  - VITE_GATEWAY_URL (fallback)

Example: VITE_EVOLUTION_API_URL=https://api.openevolve.com

The application cannot start without this configuration.
```

### Invalid URL Format
```
CRITICAL: Invalid VITE_EVOLUTION_API_URL format: "not-a-url".
URL must be a valid absolute URL including protocol (http:// or https://).
Examples:
  - http://localhost:8000 (development)
  - https://api.openevolve.com (production)
```

### Service Health Check Failed
```
OpenEvolve Integration health check failed.
Failed services:
  - Knowledge engine: Connection refused
  - Workflow orchestrator: Timeout after 30s

To bypass this validation (not recommended), set skipValidation=true.
This is only safe for testing environments.
```

---

## Testing Checklist

- [ ] Test production build with missing env vars (should crash)
- [ ] Test production build with valid env vars (should succeed)
- [ ] Test development build without env vars (should warn + use localhost)
- [ ] Test with invalid URL format (should crash with validation error)
- [ ] Test health check with services down (should fail startup)
- [ ] Test health check with services up (should log success)
- [ ] Update unit tests to use `skipValidation=true`

---

## Files Modified

1. `BubbleLab/integrations/openevolve/service-bubbles/knowledge-engine-bubble.ts`
2. `BubbleLab/apps/bubble-studio/src/env.ts`
3. `BubbleLab/integrations/openevolve/index.ts`

---

## Need Help?

See detailed report: `CRITICAL_BUG_FIXES_REPORT.md`

Federation Constitution: `CLAUDE.md`

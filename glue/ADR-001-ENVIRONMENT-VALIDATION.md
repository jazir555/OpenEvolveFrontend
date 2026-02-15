# ADR 001: Centralized Environment Variable Management

## Status
Accepted

## Context
The OpenEvolve Federation integrates 30+ independent systems, each with their own configuration needs. Historically, configuration was scattered across multiple `.env.example` files with inconsistent formatting, undocumented defaults, and no validation. This led to:

- Runtime failures due to missing environment variables
- Silent use of unintended default values
- Difficulty understanding what configuration is available
- Inconsistent validation across adapters
- Security issues with hardcoded defaults

Following the Federation Constitution's **Law of Configuration Explicitness**, we need a centralized approach where:
1. All configurable values are explicitly set via environment variables
2. Application crashes immediately if required vars are missing
3. Configuration is validated at startup with type checking

## Decision

We adopted a **centralized environment variable management system** with three pillars:

### 1. Documentation (Single Source of Truth)

**File:** `glue/ENVIRONMENT_VARIABLES.md`
- Complete registry of ALL environment variables
- For each variable:
  - Name and type
  - Required/Optional status
  - Default value (or "NONE - crashes if missing")
  - Example value
  - Which component uses it
  - Validation rules (port range, URL format, etc.)

**File:** `.env.schema`
- Template with all variables and documented defaults
- Can be copied to `.env` and filled in
- Serves as quick reference for deployment

### 2. Code Schema (TypeScript)

**File:** `glue/lib/env-schema.ts`
- Exports typed schema arrays matching documentation
- Organized by component (core, adapters, infrastructure)
- Provides `ALL_ENV_VARS` for global validation
- Provides `getSchemaForComponent(name)` for per-adapter validation

### 3. Validation Library

**File:** `glue/lib/env-validator.ts`
- `validateEnv()` - Simple presence check
- `validateEnvWithTypes()` - Full type validation
- `getEnv()` - Get single variable with type checking
- Validates:
  - **url**: Must parse as valid URL
  - **port**: Must be 1-65535
  - **number**: Must be numeric
  - **boolean**: Must be true/false or 1/0
- **Crashes immediately** with clear error message if validation fails

**File:** `glue/lib/startup-validation.ts`
- Examples of using the validator
- `validateAdapterStartup(adapterName)` - Per-adapter validation
- `validateGlobalConfig()` - Validate everything
- Custom validation examples (e.g., Graphiti with Neo4j checks)

## Consequences

### Positive

1. **No Silent Failures**: Application crashes immediately with clear error if required vars are missing
2. **Type Safety**: Ports, URLs, booleans validated at startup
3. **Single Source of Truth**: Documentation and schema stay in sync
4. **Developer Experience**: Easy to find what variables are available and how to use them
5. **Security**: No hardcoded secrets, explicit configuration required
6. **Consistency**: All adapters use same validation approach
7. **Maintainability**: Adding new var requires updating ONE place (schema + docs)

### Negative

1. **Initial Setup Overhead**: Must configure all required vars before first run (by design)
2. **Documentation Maintenance**: Adding new var requires updating THREE files:
   - `glue/ENVIRONMENT_VARIABLES.md` (docs)
   - `.env.schema` (template)
   - `glue/lib/env-schema.ts` (code)
3. **No Per-Environment Defaults**: Must explicitly set all variables (by design)

### Mitigations

1. For "Initial Setup Overhead":
   - `.env.schema` provides template with all documented defaults
   - Clear error messages guide users to fill in missing vars
   - Quick start guide in `glue/README.md`

2. For "Documentation Maintenance":
   - Schema is code-first, docs are generated
   - Could add script to auto-generate docs from schema
   - TypeScript types prevent drift

3. For "No Per-Environment Defaults":
   - Use `.env.development`, `.env.production` files
   - Docker Compose overrides for each environment
   - This is intentional: explicit configuration > magic defaults

## Implementation

### Adding a New Environment Variable

1. **Update Schema** (`glue/lib/env-schema.ts`):
   ```typescript
   export const MY_ADAPTER_ENV_VARS: EnvVar[] = [
     { name: 'MY_ADAPTER_PORT', type: 'port', required: false, default: 3000 },
     { name: 'MY_ADAPTER_API_URL', type: 'url', required: true },
     { name: 'MY_ADAPTER_API_KEY', type: 'string', required: true },
   ];
   ```

2. **Update Documentation** (`glue/ENVIRONMENT_VARIABLES.md`):
   ```markdown
   | Variable | Type | Required | Default | Description |
   |----------|------|----------|---------|-------------|
   | MY_ADAPTER_PORT | port | No | 3000 | Adapter port |
   | MY_ADAPTER_API_URL | url | **YES** | NONE | API URL |
   | MY_ADAPTER_API_KEY | string | **YES** | NONE | API key |
   ```

3. **Update Template** (`.env.schema`):
   ```bash
   MY_ADAPTER_PORT=3000
   MY_ADAPTER_API_URL=http://my-service:8000
   MY_ADAPTER_API_KEY=your-api-key-here
   ```

4. **Use in Adapter** (`my-adapter/src/index.ts`):
   ```typescript
   import { validateAdapterStartup } from '../../../lib/startup-validation';

   try {
     const config = validateAdapterStartup('myAdapter');
     // Start adapter
   } catch (error) {
     process.exit(1);
   }
   ```

### Validation Flow

```
Application Startup
    ↓
Validate Env Vars (validateEnvWithTypes)
    ↓
Type Checks (port, url, boolean, number)
    ↓
Required Var Check (crash if missing)
    ↓
Return Validated Config Object
    ↓
Start Application
```

### Error Handling

**Good Error Message:**
```
Environment validation failed:
Missing required environment variable: MY_ADAPTER_API_KEY
MY_ADAPTER_PORT: "99999" is not a valid port (1-65535)
MY_ADAPTER_API_URL: "localhost" is not a valid URL

Application cannot start. Fix the errors above and try again.
```

**Bad (What we avoid):**
```
Error: Cannot read property 'split' of undefined
    at MyAdapter.connect (/src/my-adapter.ts:42:15)
```

## Alternatives Considered

### Alternative 1: Configuration Files (YAML/JSON)
**Rejected because:**
- Requires file I/O and parsing logic
- Harder to manage in containerized environments
- Environment variables are de facto standard for containers

### Alternative 2: Distributed .env.example Files
**Rejected because:**
- Scattered across 30+ directories
- Inconsistent formatting
- No single source of truth
- Difficult to find what variables exist

### Alternative 3: Runtime Discovery
**Rejected because:**
- Violates "Law of Configuration Explicitness"
- Silent failures when services unavailable
- Hard to debug missing configuration

### Alternative 4: Service Discovery (Consul, etcd)
**Rejected because:**
- Adds infrastructure dependency
- Overkill for static configuration
- Still need initial config to connect to discovery service

## Related

- [Federation Constitution](../../CLAUDE.md) - Law of Configuration Explicitness
- [Environment Variable Registry](./ENVIRONMENT_VARIABLES.md) - Complete documentation
- [Environment Schema](../../.env.schema) - Configuration template
- [Env Validator](./lib/env-validator.ts) - Validation library
- [Startup Validation Examples](./lib/startup-validation.ts) - Usage examples

## References

- [The Twelve-Factor App: Config](https://12factor.net/config)
- [Environment Variables](https://en.wikipedia.org/wiki/Environment_variable)
- [Container Configuration Best Practices](https://kubernetes.io/docs/tasks/inject-data-application/define-environment-variable-container/)

## History

- **2025-01-XX**: Initial decision to centralize environment variables
- **2025-01-XX**: Implemented schema, validator, and documentation
- **2025-01-XX**: Accepted as standard approach for all adapters

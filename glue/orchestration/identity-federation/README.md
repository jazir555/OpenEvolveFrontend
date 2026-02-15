# Identity Federation

Centralized authentication and user management for OpenEvolve services.

## Overview

This module implements the Identity Federation Strategy (ADR-006) with three phases:

1. **OIDC First** - Direct OpenID Connect integration
2. **Header Injection** - OAuth2-Proxy fallback for services without OIDC
3. **Shadow Account Sync** - Idempotent user synchronization for legacy services

## Components

### OIDCProvider
Direct integration with OpenID Connect providers (Keycloak, Auth0, Okta, etc.).

**Features:**
- Fetch provider configuration from `.well-known/openid-configuration`
- Generate authorization URLs
- Exchange authorization codes for tokens
- Refresh access tokens
- Fetch user info from UserInfo endpoint
- Validate ID tokens
- Generate logout URLs

**Usage:**
```typescript
import { OIDCProvider } from '@openevolve/identity-federation';

const oidc = new OIDCProvider({
  issuer: 'https://keycloak.example.com/realms/myrealm',
  clientId: 'my-client',
  clientSecret: 'my-secret',
  redirectUri: 'http://localhost:3000/callback',
});

// Get login URL
const authUrl = await oidc.getAuthorizationUrl();

// Exchange code for tokens
const tokens = await oidc.exchangeCodeForTokens(code);

// Get user info
const userInfo = await oidc.getUserInfo(tokens.access_token);
```

### HeaderInjectionAuth
OAuth2-Proxy pattern for services that don't support OIDC natively.

**Features:**
- Extract user information from injected headers
- Validate authentication headers
- Parse groups/roles
- Middleware for Express-like frameworks
- Cookie validation
- Domain whitelisting

**Usage:**
```typescript
import { HeaderInjectionAuth } from '@openevolve/identity-federation';

const headerAuth = new HeaderInjectionAuth({
  requireAuth: true,
  userHeader: 'X-Remote-User',
  emailHeader: 'X-Remote-Email',
  groupsHeader: 'X-Remote-Groups',
});

// Express middleware
app.use(headerAuth.createMiddleware(['admin', 'users']));

// Manual validation
if (headerAuth.validateHeaders(request.headers)) {
  const user = headerAuth.extractUserFromHeaders(request.headers);
}
```

### ShadowAccountSync
Idempotent user synchronization for services requiring local accounts.

**Features:**
- Idempotent sync operations (safe to run multiple times)
- Batch user synchronization
- Update existing accounts (don't create duplicates)
- Stale account cleanup
- Circuit breaker protection
- Retry with exponential backoff
- Dry-run mode for testing

**Usage:**
```typescript
import { ShadowAccountSync } from '@openevolve/identity-federation';

const sync = new ShadowAccountSync();

// Register service adapter
sync.registerService('graphiti', new GraphitiAdapter());

// Sync users
const result = await sync.syncUsers('graphiti', centralUsers, {
  dryRun: false,
  batchSize: 50,
  continueOnError: true,
});

// Cleanup stale accounts
await sync.cleanupStaleAccounts('graphiti', 90 * 24 * 60 * 60 * 1000);
```

## Configuration

### Environment Variables

```bash
# OIDC Configuration
OIDC_ISSUER=https://keycloak.example.com/realms/myrealm
OIDC_CLIENT_ID=my-client
OIDC_CLIENT_SECRET=my-secret
OIDC_REDIRECT_URI=http://localhost:3000/callback
OIDC_SCOPES=openid,profile,email

# OAuth2-Proxy Configuration
OAUTH2_PROXY_COOKIE_SECRET=your-secret-key
OAUTH2_PROXY_COOKIE_NAME=_oauth2_proxy

# User Sync Configuration
USER_SYNC_BATCH_SIZE=50
USER_SYNC_STALE_THRESHOLD_DAYS=90
```

## Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │ Login
       ▼
┌─────────────────┐
│  Central IdP    │  (Keycloak, Auth0, etc.)
│  (OIDC Provider)│
└────────┬────────┘
         │ JWT Token
         ▼
┌─────────────────────────────────────┐
│        Frontend Application         │
│  (Validates token, gets user info)  │
└────────┬────────────────────────────┘
         │
    ┌────┴─────┐
    │  Auth    │
    │ Sidecar  │  (OAuth2-Proxy)
    └────┬─────┘
         │ X-Remote-User Headers
         ▼
┌─────────────────────────────────────┐
│        All Backend Services         │
│  (Trust headers, sync shadow accounts)│
└─────────────────────────────────────┘
```

## Implementation Strategy

### Phase 1: OIDC First (Preferred)
1. Configure services to trust Central IdP
2. Services validate JWT tokens directly
3. No account sync needed - all use IdP

### Phase 2: Header Injection (Fallback)
1. Deploy OAuth2-Proxy sidecar
2. Sidecar validates JWT
3. Sidecar injects user headers
4. Services trust headers (network isolation)

### Phase 3: Shadow Account Sync (Last Resort)
1. Run sync script on first login
2. Script creates/updates local account
3. Idempotent - safe to run multiple times
4. Sync updates when user changes in IdP

## Gotchas

### Gotcha 1: Token Expiration
JWT tokens expire (typically 1 hour). Frontend must:
- Refresh tokens before expiry
- Handle 401 responses
- Use refresh tokens (not access tokens)

### Gotcha 2: Group Mapping
IdP groups may not match service roles. Need mapping:
```typescript
const groupMapping = {
  'openevolve-admins': 'admin',
  'openevolve-users': 'user',
  'openevolve-read-only': 'viewer'
};
```

### Gotcha 3: Service-Specific Requirements
Some services require additional user attributes. Must:
- Add custom claims to JWT
- Store additional attributes in user profile
- Sync these attributes to shadow accounts

## Testing

### Unit Tests
```bash
npm run test
```

### Integration Tests
```bash
# Test OIDC flow
OIDC_ISSUER=https://keycloak.test.local npm test

# Test header injection
OAUTH2_PROXY_COOKIE_SECRET=test npm test

# Test user sync
USER_SYNC_DRY_RUN=true npm test
```

## References
- [ADR-006: Identity Federation Strategy](../../docs/adrs/006-identity-federation.md)
- [Federation Constitution](../../README.md)
- [OIDC Specification](https://openid.net/specs/openid-connect-core-1_0.html)

# ADR-006: Identity Federation Strategy

## Status
**Proposed**

## Context
The OpenEvolve Frontend integrates 30+ services, each potentially requiring authentication:
- OpenEvolve API
- RAGBits
- Datapizza
- Z3 Solver
- LeanAide
- And 25+ more...

**The Problem**: How do we authenticate users across all these services without:
- Requiring users to log in 30 times?
- Storing 30 different sets of credentials?
- Managing 30 different user directories?

## Decision
Implement **Centralized Identity Federation** following the Federation Constitution.

### Architecture

#### Strategy: OIDC First with Header Injection Fallback

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
┌─────────────────┐
│  Frontend App   │
│  (BubbleLab)    │
└────────┬────────┘
         │
    ┌────┴─────┐
    │  Auth    │
    │ Sidecar  │  (OAuth2-Proxy)
    └────┬─────┘
         │ X-Remote-User Headers
         ▼
┌─────────────────────────────────────┐
│        All Backend Services         │
│  (OpenEvolve, RAGBits, Datapizza)  │
│  Trust Central IdP or Headers       │
└─────────────────────────────────────┘
```

### Implementation

#### Phase 1: OIDC First (Preferred)
1. Configure all core services to trust Central IdP
2. Services validate JWT tokens directly
3. No account sync needed - all use IdP

**Benefits**:
- Clean architecture
- No shadow accounts
- Single source of truth

**Requirements**:
- All services support OIDC
- Can configure JWT verification

#### Phase 2: Header Injection (Fallback)
1. Deploy OAuth2-Proxy sidecar
2. Sidecar validates JWT
3. Sidecar injects headers:
   - `X-Remote-User`: username
   - `X-Remote-Email`: email
   - `X-Remote-Groups`: groups
4. Services trust headers (only from sidecar)

**Benefits**:
- Works with services that don't support OIDC
- Minimal service changes

**Risks**:
- Services must only trust sidecar (network isolation required)

#### Phase 3: Shadow Account Sync (Last Resort)
For services that require local user accounts:

```typescript
// Idempotent user sync script
async function syncUserToService(user: User, service: Service) {
  // Check if shadow account exists
  const existing = await service.getUserByRemoteId(user.id);

  if (existing) {
    // Update if changed (idempotent)
    if (existing.email !== user.email) {
      await service.updateUser(existing.id, { email: user.email });
    }
    return existing;
  }

  // Create new shadow account
  return await service.createUser({
    remote_id: user.id,
    username: user.username,
    email: user.email,
    groups: user.groups
  });
}
```

**Requirements**:
- Run sync on first login
- Make it idempotent (safe to run multiple times)
- Sync updates when user changes in IdP

### Gotchas

#### Gotcha 1: Token Expiration
JWT tokens expire (typically 1 hour). Frontend must:
- Refresh tokens before expiry
- Handle 401 responses by re-authenticating
- Use refresh tokens (not access tokens) for refresh

#### Gotcha 2: Group Mapping
IdP groups may not match service roles. Need mapping:
```typescript
const groupMapping = {
  'openevolve-admins': 'admin',
  'openevolve-users': 'user',
  'openevolve-read-only': 'viewer'
};
```

#### Gotcha 3: Service-Specific Requirements
Some services require additional user attributes. Must:
- Add custom claims to JWT
- Store additional attributes in user profile
- Sync these attributes to shadow accounts

## Consequences

### Positive
- ✅ **Single login**: Users authenticate once
- ✅ **Centralized**: User management in one place
- ✅ **Secure**: Industry-standard protocols
- ✅ **Flexible**: Works with various IdP providers

### Negative
- ⚠️ **Complexity**: Additional infrastructure (IdP, sidecars)
- ⚠️ **Dependencies**: All services depend on IdP availability
- ⚠️ **Migration**: Need to migrate existing users

### Mitigations
- Use high-availability IdP setup
- Implement fallback authentication for critical services
- Plan migration carefully with user communication

## Alternatives Considered

### Alternative 1: Per-Service Authentication
**Description**: Each service manages its own users

**Pros**: Simple, no dependencies

**Cons**: 30 logins, UX nightmare, credential sprawl

**Rejected**: Unacceptable user experience

### Alternative 2: Shared Database
**Description**: All services share a user database

**Pros**: Single user store

**Cons**: Tight coupling, database becomes bottleneck, security risk

**Rejected**: Violates isolation requirements

### Alternative 3: API Gateway Authentication
**Description**: Gateway handles auth, passes user context to services

**Pros**: Centralized auth logic

**Cons**: Gateway becomes bottleneck, services still need auth

**Rejected**: Just moves the problem

## Related Decisions
- [ADR-005: Anti-Corruption Layer](./005-acl.md)

## Implementation Date
2026-02-15 (Proposed)
2026-03-01 (Target Implementation)

## Author
OpenEvolve Federation Team

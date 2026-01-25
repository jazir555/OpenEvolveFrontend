# Authentication & Authorization

Complete guide for authentication and authorization in BubbleLab.

**Table of Contents:**
- [Overview](#overview)
- [Authentication Methods](#authentication-methods)
- [Authorization](#authorization)
- [Credential Management](#credential-management)
- [Security Best Practices](#security-best-practices)
- [Token Management](#token-management)
- [Session Management](#session-management)
- [OAuth 2.0 Flows](#oauth-20-flows)
- [Multi-Factor Authentication](#multi-factor-authentication)
- [Troubleshooting](#troubleshooting)

---

## Overview

BubbleLab provides comprehensive authentication and authorization capabilities:

- **Multiple Authentication Methods**: API keys, OAuth 2.0, JWT, Basic Auth
- **Secure Credential Storage**: Encrypted storage with access controls
- **Role-Based Access Control (RBAC)**: Granular permissions
- **Session Management**: Secure session handling
- **Token Management**: JWT token generation and validation
- **Multi-Factor Authentication (MFA)**: Optional 2FA support

### Security Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Security Layer                       │
├─────────────────────────────────────────────────────────┤
│  Authentication                                         │
│  ├── API Keys                                          │
│  ├── OAuth 2.0                                         │
│  ├── JWT Tokens                                        │
│  └── Basic Auth                                        │
├─────────────────────────────────────────────────────────┤
│  Authorization                                         │
│  ├── Role-Based Access Control                         │
│  ├── Permission Scopes                                 │
│  └── Resource-Level Permissions                        │
├─────────────────────────────────────────────────────────┤
│  Credential Management                                  │
│  ├── Encrypted Storage                                 │
│  ├── Key Rotation                                      │
│  └── Access Auditing                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Authentication Methods

### API Key Authentication

**Overview**: Simple authentication using API keys

**Use Cases:**
- Service-to-service authentication
- Simple integrations
- Testing and development

**Setup:**

```typescript
// Generate API key in BubbleLab dashboard
// Settings > API Keys > Generate New Key

// Store securely
const credentials = {
  apiKey: process.env.BUBBLELAB_API_KEY,
  // Or use credential management system
};

// Use in requests
const result = await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/flows',
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${credentials.apiKey}`,
    'X-API-Key': credentials.apiKey
  }
});
```

**API Key Format:**

```
bl_live_51AbC...     // Live environment key
bl_test_51AbC...    // Test environment key
bl_secret_51AbC...  // Secret key (more privileges)
```

**Best Practices:**
- Use environment-specific keys
- Rotate keys regularly (90 days recommended)
- Revoke compromised keys immediately
- Use minimum required permissions
- Never commit keys to version control

**Security Features:**
- Key prefixing for easy identification
- Automatic expiration options
- IP whitelisting support
- Usage rate limiting
- Audit logging

---

### OAuth 2.0 Authentication

**Overview**: Industry-standard authorization framework

**Supported Flows:**
- Authorization Code Flow (for server-side apps)
- Implicit Flow (for client-side apps)
- Client Credentials Flow (for service-to-service)
- Device Code Flow (for IoT/headless devices)

#### Authorization Code Flow

**Use Case**: Server-side applications with secure backend

**Setup:**

```typescript
// 1. Register application
const clientId = 'your-client-id';
const clientSecret = 'your-client-secret';
const redirectUri = 'https://your-app.com/callback';

// 2. Redirect user to authorization URL
const authUrl = `https://auth.bubblelab.io/oauth/authorize?` +
  `response_type=code&` +
  `client_id=${clientId}&` +
  `redirect_uri=${encodeURIComponent(redirectUri)}&` +
  `scope=flows:read flows:write&` +
  `state=${generateState()}`;

// 3. User authorizes, receives callback with code
// 4. Exchange code for access token
const tokenResponse = await httpBubble.execute({
  url: 'https://auth.bubblelab.io/oauth/token',
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: new URLSearchParams({
    grant_type: 'authorization_code',
    code: authCode,
    client_id: clientId,
    client_secret: clientSecret,
    redirect_uri: redirectUri
  })
});

const { access_token, refresh_token, expires_in } = tokenResponse.data;

// 5. Use access token
const result = await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/flows',
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${access_token}`
  }
});
```

**Token Refresh:**

```typescript
// Refresh access token
const refreshResponse = await httpBubble.execute({
  url: 'https://auth.bubblelab.io/oauth/token',
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: new URLSearchParams({
    grant_type: 'refresh_token',
    refresh_token: refresh_token,
    client_id: clientId,
    client_secret: clientSecret
  })
});

const { access_token: newAccessToken } = refreshResponse.data;
```

---

#### Client Credentials Flow

**Use Case**: Service-to-service authentication

**Setup:**

```typescript
const tokenResponse = await httpBubble.execute({
  url: 'https://auth.bubblelab.io/oauth/token',
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: new URLSearchParams({
    grant_type: 'client_credentials',
    client_id: clientId,
    client_secret: clientSecret,
    scope: 'flows:read flows:write'
  })
});

const { access_token, expires_in } = tokenResponse.data;
```

---

### JWT Token Authentication

**Overview**: Stateless authentication using JSON Web Tokens

**Use Cases:**
- Distributed systems
- Mobile applications
- Microservices architecture

**Token Structure:**

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user-123",
    "name": "John Doe",
    "email": "john@example.com",
    "roles": ["user", "admin"],
    "scopes": ["flows:read", "flows:write"],
    "iat": 1642579200,
    "exp": 1642582800
  },
  "signature": "..."
}
```

**Generation:**

```typescript
const jwt = require('jsonwebtoken');

// Generate JWT token
const token = jwt.sign(
  {
    sub: 'user-123',
    name: 'John Doe',
    email: 'john@example.com',
    roles: ['user', 'admin'],
    scopes: ['flows:read', 'flows:write']
  },
  process.env.JWT_SECRET,
  {
    expiresIn: '1h',
    issuer: 'bubblelab',
    audience: 'bubblelab-api'
  }
);

// Use in requests
const result = await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/flows',
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

**Validation:**

```typescript
// Validate JWT token
function validateToken(token) {
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET, {
      issuer: 'bubblelab',
      audience: 'bubblelab-api'
    });

    return {
      valid: true,
      user: decoded
    };
  } catch (error) {
    return {
      valid: false,
      error: error.message
    };
  }
}

// Usage
const validation = validateToken(request.headers.authorization);

if (!validation.valid) {
  return {
    status: 401,
    error: 'Invalid token'
  };
}

// User is authenticated
const user = validation.user;
```

---

### Basic Authentication

**Overview**: Simple username/password authentication

**Use Cases:**
- Simple integrations
- Legacy system compatibility
- Testing environments

**Setup:**

```typescript
const credentials = {
  username: 'user@example.com',
  password: 'password123'
};

// Encode to base64
const auth = Buffer.from(
  `${credentials.username}:${credentials.password}`
).toString('base64');

// Use in requests
const result = await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/flows',
  method: 'GET',
  headers: {
    'Authorization': `Basic ${auth}`
  }
});
```

**Best Practices:**
- Only use over HTTPS
- Rotate credentials regularly
- Use strong passwords
- Consider API keys instead for production

---

## Authorization

### Role-Based Access Control (RBAC)

**Overview**: Access control based on user roles

**Built-in Roles:**

| Role | Permissions | Description |
|------|-------------|-------------|
| `admin` | All permissions | Full system access |
| `owner` | All permissions | Workspace owner |
| `developer` | flows:read, flows:write, runs:read, runs:execute | Development access |
| `viewer` | flows:read, runs:read | Read-only access |
| `executor` | runs:read, runs:execute | Execute-only access |

**Assigning Roles:**

```typescript
// Assign role to user
await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/users/${userId}/roles`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${adminToken}`
  },
  body: {
    role: 'developer',
    scope: 'workspace-123'
  }
});
```

**Checking Permissions:**

```typescript
function hasPermission(user, requiredPermission) {
  return user.roles.some(role => {
    const rolePermissions = ROLE_PERMISSIONS[role];
    return rolePermissions.includes(requiredPermission);
  });
}

// Usage
if (!hasPermission(user, 'flows:write')) {
  throw new Error('Insufficient permissions');
}
```

---

### Permission Scopes

**Available Scopes:**

| Scope | Description |
|-------|-------------|
| `flows:read` | Read flows |
| `flows:write` | Create and modify flows |
| `flows:delete` | Delete flows |
| `runs:read` | Read run history |
| `runs:execute` | Execute flows |
| `runs:cancel` | Cancel running flows |
| `credentials:read` | Read credentials |
| `credentials:write` | Manage credentials |
| `users:read` | Read users |
| `users:write` | Manage users |
| `admin:all` | All administrative permissions |

**Using Scopes:**

```typescript
// Request specific scopes during OAuth flow
const authUrl = `https://auth.bubblelab.io/oauth/authorize?` +
  `client_id=${clientId}&` +
  `scope=${encodeURIComponent('flows:read flows:write runs:execute')}&` +
  `response_type=code`;

// Check scopes in JWT
function hasScope(token, requiredScope) {
  const decoded = jwt.decode(token);
  return decoded.scopes.includes(requiredScope);
}
```

---

### Resource-Level Permissions

**Overview**: Fine-grained permissions on specific resources

**Example:**

```typescript
// Grant user access to specific flow
await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/flows/${flowId}/permissions`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${adminToken}`
  },
  body: {
    userId: 'user-456',
    permission: 'write'
  }
});

// Check permissions
const permissions = await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/flows/${flowId}/permissions`,
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${userToken}`
  }
});

if (permissions.data.permission === 'write') {
  // User can write to this flow
}
```

**Permission Levels:**
- `none`: No access
- `read`: Read-only access
- `write`: Read and write access
- `admin`: Full control (including permissions management)

---

## Credential Management

### Secure Storage

**Overview**: Encrypted credential storage

**Features:**
- AES-256 encryption
- Key rotation support
- Access logging
- Automatic expiration

**Storing Credentials:**

```typescript
// Store credential
await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/credentials',
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: {
    name: 'OpenAI API Key',
    type: 'api_key',
    value: 'sk-...', // Will be encrypted
    expiresAt: '2024-12-31T23:59:59Z'
  }
});
```

**Retrieving Credentials:**

```typescript
// Retrieve credential (decrypted)
const credential = await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/credentials/${credentialId}`,
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

const value = credential.data.value; // Decrypted value
```

---

### Key Rotation

**Overview**: Periodic rotation of credentials

**Automated Rotation:**

```typescript
// Schedule key rotation
await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/credentials/${credentialId}/rotation`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: {
    schedule: '0 0 * * 0', // Every Sunday at midnight
    notifyDaysBefore: 7
  }
});
```

**Manual Rotation:**

```typescript
// Rotate credential
const newCredential = await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/credentials/${credentialId}/rotate`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

const newKeyValue = newCredential.data.value;
```

---

### Access Auditing

**Overview**: Track credential access

**View Access Logs:**

```typescript
// Get access logs for credential
const logs = await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/credentials/${credentialId}/audit`,
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  params: {
    from: '2024-01-01',
    to: '2024-01-31'
  }
});

logs.data.forEach(log => {
  console.log(`${log.timestamp} - ${log.user} - ${log.action}`);
});
```

---

## Security Best Practices

### Password Security

**Requirements:**
- Minimum 12 characters
- Uppercase and lowercase letters
- At least one number
- At least one special character
- No common words or patterns

**Hashing:**

```typescript
const bcrypt = require('bcrypt');

// Hash password
const hash = await bcrypt.hash(password, 10);

// Verify password
const isValid = await bcrypt.compare(password, hash);
```

---

### API Key Security

**Best Practices:**

1. **Use environment variables**
```typescript
const apiKey = process.env.BUBBLELAB_API_KEY;
```

2. **Never hardcode keys**
```typescript
// BAD
const apiKey = 'bl_live_51AbC...';

// GOOD
const apiKey = process.env.BUBBLELAB_API_KEY;
```

3. **Use key prefixes for identification**
```typescript
// Production key
const prodKey = 'bl_live_...';

// Test key
const testKey = 'bl_test_...';

// Secret key
const secretKey = 'bl_secret_...';
```

4. **Rotate keys regularly**
```typescript
// Check key age
const keyAge = Date.now() - keyCreatedAt;
if (keyAge > 90 * 24 * 60 * 60 * 1000) { // 90 days
  // Rotate key
}
```

5. **Revoke compromised keys immediately**
```typescript
await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/keys/${keyId}/revoke`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${adminToken}`
  }
});
```

---

### Token Security

**Best Practices:**

1. **Set appropriate expiration**
```typescript
const token = jwt.sign(payload, secret, {
  expiresIn: '1h' // Short-lived access tokens
});

const refreshToken = jwt.sign(payload, secret, {
  expiresIn: '30d' // Long-lived refresh tokens
});
```

2. **Use HTTPS only**
```typescript
// Redirect HTTP to HTTPS
if (req.protocol === 'http') {
  return res.redirect(301, `https://${req.headers.host}${req.url}`);
}
```

3. **Store tokens securely**
```typescript
// Use HttpOnly cookies for web apps
res.cookie('token', token, {
  httpOnly: true,
  secure: true,
  sameSite: 'strict',
  maxAge: 3600000 // 1 hour
});

// Use secure storage for mobile apps
// iOS: Keychain
// Android: Keystore
```

4. **Implement token revocation**
```typescript
// Revoke token
await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/tokens/revoke`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: {
    token: token
  }
});
```

---

## Token Management

### Token Lifecycle

```
Issuance → Usage → Refresh → Expiration → Revocation
    ↑                                      |
    |______________________________________|
                  (optional loop)
```

### Token Types

| Type | Lifetime | Use Case | Storage |
|------|----------|----------|---------|
| Access Token | 1 hour | API requests | Memory/HttpOnly cookie |
| Refresh Token | 30 days | Get new access tokens | Secure storage |
| ID Token | 1 hour | User identity | Memory |
| API Key | No expiration | Service authentication | Environment variable |

### Token Refresh Strategy

```typescript
class TokenManager {
  constructor() {
    this.accessToken = null;
    this.refreshToken = null;
    this.tokenExpiresAt = null;
  }

  async getAccessToken() {
    // Return valid token if available
    if (this.accessToken && Date.now() < this.tokenExpiresAt) {
      return this.accessToken;
    }

    // Refresh token
    return await this.refreshAccessToken();
  }

  async refreshAccessToken() {
    const response = await httpBubble.execute({
      url: 'https://auth.bubblelab.io/oauth/token',
      method: 'POST',
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: this.refreshToken
      })
    });

    this.accessToken = response.data.access_token;
    this.refreshToken = response.data.refresh_token;
    this.tokenExpiresAt = Date.now() + (response.data.expires_in * 1000);

    return this.accessToken;
  }

  async initialize(username, password) {
    const response = await httpBubble.execute({
      url: 'https://auth.bubblelab.io/oauth/token',
      method: 'POST',
      body: new URLSearchParams({
        grant_type: 'password',
        username: username,
        password: password
      })
    });

    this.accessToken = response.data.access_token;
    this.refreshToken = response.data.refresh_token;
    this.tokenExpiresAt = Date.now() + (response.data.expires_in * 1000);
  }
}

// Usage
const tokenManager = new TokenManager();
await tokenManager.initialize('user@example.com', 'password');

const token = await tokenManager.getAccessToken();
```

---

## Session Management

### Session Lifecycle

```
Login → Session Creation → Usage → Refresh → Logout → Session Deletion
```

### Creating Sessions

```typescript
// Create session
const session = await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/sessions',
  method: 'POST',
  body: {
    email: 'user@example.com',
    password: 'password123',
    deviceInfo: {
      userAgent: req.headers['user-agent'],
      ip: req.ip
    }
  }
});

const { sessionId, accessToken, refreshToken } = session.data;
```

### Managing Sessions

```typescript
// List active sessions
const sessions = await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/sessions',
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

// Revoke session
await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/sessions/${sessionId}/revoke`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

// Revoke all sessions (logout everywhere)
await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/sessions/revoke-all',
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

---

## OAuth 2.0 Flows

### Authorization Code Flow with PKCE

**Enhanced security for public clients**

```typescript
// Generate code verifier and challenge
const crypto = require('crypto');

function generateCodeVerifier() {
  return crypto.randomBytes(32).toString('base64url');
}

function generateCodeChallenge(verifier) {
  return crypto.createHash('sha256').update(verifier).digest('base64url');
}

const codeVerifier = generateCodeVerifier();
const codeChallenge = generateCodeChallenge(codeVerifier);

// Store verifier for later use (session storage)

// 1. Redirect to authorization URL
const authUrl = `https://auth.bubblelab.io/oauth/authorize?` +
  `response_type=code&` +
  `client_id=${clientId}&` +
  `redirect_uri=${encodeURIComponent(redirectUri)}&` +
  `code_challenge=${codeChallenge}&` +
  `code_challenge_method=S256`;

// 2. User authorizes, receives code

// 3. Exchange code for token (with verifier)
const tokenResponse = await httpBubble.execute({
  url: 'https://auth.bubblelab.io/oauth/token',
  method: 'POST',
  body: new URLSearchParams({
    grant_type: 'authorization_code',
    code: authCode,
    client_id: clientId,
    redirect_uri: redirectUri,
    code_verifier: codeVerifier
  })
});
```

---

## Multi-Factor Authentication

### Enabling MFA

```typescript
// Enable MFA for user
await httpBubble.execute({
  url: `https://api.bubblelab.io/v1/users/${userId}/mfa`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: {
    type: 'totp' // Time-based one-time password
  }
});

// Returns QR code for setup
const { qrCodeUrl, secret } = response.data;
```

### Verifying MFA

```typescript
// Login with MFA
const loginResponse = await httpBubble.execute({
  url: 'https://api.bubblelab.io/v1/auth/login',
  method: 'POST',
  body: {
    email: 'user@example.com',
    password: 'password123'
  }
});

// Requires MFA verification
if (loginResponse.data.requiresMfa) {
  const mfaResponse = await httpBubble.execute({
    url: 'https://api.bubblelab.io/v1/auth/mfa/verify',
    method: 'POST',
    body: {
      sessionId: loginResponse.data.sessionId,
      code: '123456' // From authenticator app
    }
  });

  const { accessToken } = mfaResponse.data;
}
```

---

## Troubleshooting

### Common Issues

**Issue: Invalid API key**

```
Error: 401 Unauthorized
Solution: Verify API key is correct and not expired
```

**Issue: Token expired**

```
Error: 401 Unauthorized - Token expired
Solution: Use refresh token to get new access token
```

**Issue: Insufficient permissions**

```
Error: 403 Forbidden
Solution: Check user roles and scopes
```

**Issue: Invalid redirect URI**

```
Error: redirect_uri_mismatch
Solution: Ensure redirect URI matches registered URI
```

### Debug Mode

```typescript
// Enable debug logging
process.env.DEBUG = 'bubblelab:auth';

// Authentication requests will be logged
```

---

**Last Updated:** 2026-01-18
**Version:** 1.0.0
**Maintained By:** BubbleLab Core Team

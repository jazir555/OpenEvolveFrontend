# Environment Variable Configuration - Complete Setup Guide

This guide provides everything you need to configure environment variables for the OpenEvolve Federation.

## Quick Start (5 Minutes)

### 1. Copy the Schema
```bash
cp .env.schema .env
```

### 2. Generate Secrets
```bash
# Generate application secret
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Or if you need JWT_SECRET
JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
```

### 3. Fill in Required Variables

Edit `.env` and set these **required** variables:

```bash
# Security (REQUIRED)
SECRET_KEY=your-generated-secret-here

# Database (if using PostgreSQL)
DB_PASSWORD=your-db-password-here

# Neo4j (if using Graphiti)
NEO4J_PASSWORD=your-neo4j-password-here

# OpenAI (if using LLM features)
OPENAI_API_KEY=sk-your-openai-key-here

# Adapters (set only what you use)
BUBBLELAB_API_URL=http://bubblelab-core:8000
BUBBLELAB_API_KEY=your-bubblelab-key-here
```

### 4. Verify Configuration
```bash
# The application will validate on startup
# If there are errors, you'll see:
# Environment validation failed:
# Missing required environment variable: SECRET_KEY
```

## Documentation Files

| File | Purpose | Location |
|------|---------|----------|
| **Complete Registry** | All variables documented with types, defaults, validation | `glue/ENVIRONMENT_VARIABLES.md` |
| **Schema Template** | Copy this to `.env` and fill in | `.env.schema` |
| **Setup Guide** | This file - quick start and troubleshooting | `ENVIRONMENT_SETUP_GUIDE.md` |
| **TypeScript Schema** | Code schema for programmatic validation | `glue/lib/env-schema.ts` |
| **Validation Library** | Functions to validate at startup | `glue/lib/env-validator.ts` |
| **Usage Examples** | How to use validation in your code | `glue/lib/startup-validation.ts` |
| **Architecture Decision** | Why we chose this approach | `glue/ADR-001-ENVIRONMENT-VALIDATION.md` |

## Required Variables by Use Case

### Minimal Setup (No Adapters)
```bash
SECRET_KEY=your-secret
```

### With Neo4j/Graphiti
```bash
SECRET_KEY=your-secret
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

### With Knowledge Engine
```bash
SECRET_KEY=your-secret
LLM_API_KEY=your-openai-key
JWT_SECRET=your-jwt-secret
```

### With BubbleLab Adapter
```bash
SECRET_KEY=your-secret
BUBBLELAB_API_URL=http://bubblelab:8000
BUBBLELAB_API_KEY=your-key
```

### Full Production Setup
See `.env.schema` for all variables.

## Common Configuration Patterns

### Development Environment
```bash
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
OPENEVOLVE_GRAPHQL__ENABLE_PLAYGROUND=true
```

### Production Environment
```bash
DEBUG=false
LOG_LEVEL=INFO
RELOAD=false
OPENEVOLVE_GRAPHQL__ENABLE_PLAYGROUND=false
# Use strong SECRET_KEY generated with secrets.token_hex(64)
```

### Docker Compose
```yaml
services:
  api:
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=${DATABASE_URL}
    env_file:
      - .env
```

### Kubernetes
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: openevolve-secrets
type: Opaque
stringData:
  SECRET_KEY: your-generated-secret
  NEO4J_PASSWORD: your-password

---
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: api
    env:
      - name: SECRET_KEY
        valueFrom:
          secretKeyRef:
            name: openevolve-secrets
            key: SECRET_KEY
```

## Validation

### Automatic Validation (Recommended)

Application validates automatically on startup. If validation fails:

```
Environment validation failed:
Missing required environment variable: SECRET_KEY
DATABASE_URL: "invalid-url" is not a valid URL
API_PORT: "99999" is not a valid port (1-65535)

Application cannot start. Please fix the errors above.
```

### Manual Validation (Optional)

```bash
# Check if file exists
ls -la .env

# Verify syntax (no trailing spaces, etc.)
cat .env | grep -E "^\s+#" | head -20

# Try loading in shell
set -a
source .env
set +a
echo $SECRET_KEY  # Should show your value
```

## Troubleshooting

### "Application Crashes Immediately"

**Cause:** Missing required environment variable

**Solution:**
1. Check error message for which variable is missing
2. Add it to `.env`
3. Restart application

```bash
# Example error:
# Missing required environment variable: NEO4J_PASSWORD

# Solution: Add to .env
NEO4J_PASSWORD=your-password
```

### "Invalid Port"

**Cause:** Port number outside valid range (1-65535)

**Solution:**
```bash
# Wrong
API_PORT=99999

# Correct
API_PORT=8000
```

### "Invalid URL"

**Cause:** URL doesn't include protocol or is malformed

**Solution:**
```bash
# Wrong
NEO4J_URI=localhost:7687
API_URL=example.com

# Correct
NEO4J_URI=bolt://localhost:7687
API_URL=http://example.com
```

### "Connection Refused"

**Cause:** Wrong URL format for Docker or service not running

**Solution:**
```bash
# For Docker, use service name not localhost
BUBBLELAB_API_URL=http://bubblelab-core:8000  # Correct for Docker
BUBBLELAB_API_URL=http://localhost:8000        # Wrong for Docker

# Verify service is running
docker ps | grep bubblelab
```

### Variable Not Being Read

**Cause:** File format issues (spaces, quotes, etc.)

**Solution:**
```bash
# No quotes around values
SECRET_KEY="value"     # Wrong - includes quotes in value
SECRET_KEY=value         # Correct

# No spaces around =
SECRET_KEY =value       # Wrong
SECRET_KEY= value       # Wrong
SECRET_KEY=value        # Correct

# No trailing comments on value line
SECRET_KEY=value # comment  # Wrong - includes comment in value
SECRET_KEY=value          # Correct
# This is a comment       # Put comments on separate line
```

## Secret Generation

### Application Secret / JWT Secret
```bash
# 32 bytes (256 bits) - recommended
python -c "import secrets; print(secrets.token_hex(32))"

# 64 bytes (512 bits) - extra secure
python -c "import secrets; print(secrets.token_hex(64))"
```

### Random API Key
```bash
# Hex format
openssl rand -hex 32

# Base64 format
openssl rand -base64 32

# UUID format
uuidgen
```

### Database Password
```bash
# Strong password (16 chars)
openssl rand -base64 12

# Or use password manager
```

## Security Best Practices

### 1. Never Commit `.env`
```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
echo ".env.*.secret" >> .gitignore
```

### 2. Use Environment-Specific Files
```bash
# .env.development
DEBUG=true
SECRET_KEY=dev-secret-not-secure

# .env.production
DEBUG=false
SECRET_KEY=${PROD_SECRET}  # From secret manager

# .env.test
DEBUG=true
SECRET_KEY=test-secret
```

### 3. Use Secret Management (Production)

**Docker Secrets:**
```yaml
services:
  api:
    secrets:
      - secret_key
    environment:
      - SECRET_KEY_FILE=/run/secrets/secret_key
secrets:
  secret_key:
    external: true
```

**Kubernetes Secrets:**
```yaml
env:
  - name: SECRET_KEY
    valueFrom:
      secretKeyRef:
        name: openevolve-secrets
        key: secret-key
```

**AWS Secrets Manager:**
```bash
# Store secret
aws secretsmanager create-secret \
  --name openevolve/secret-key \
  --secret-string "$(python -c "import secrets; print(secrets.token_hex(32))")"

# Retrieve in application
SECRET_KEY=$(aws secretsmanager get-secret-value \
  --secret-id openevolve/secret-key \
  --query SecretString --output text)
```

## Quick Reference

### Generate All Secrets at Once
```bash
# Save this script as generate-secrets.sh
#!/bin/bash
cat > .env << EOF
# Generated on $(date)
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
NEO4J_PASSWORD=$(openssl rand -base64 12)
DB_PASSWORD=$(openssl rand -base64 12)
EOF

echo "Secrets generated in .env"
echo "DO NOT COMMIT .env TO VERSION CONTROL"
```

### Check Your Configuration
```bash
# List all set variables (excluding empty)
grep -E "^[^#].*=" .env | cut -d= -f1

# Count required variables
grep -c "REQUIRED" glue/ENVIRONMENT_VARIABLES.md

# Check for unset required vars in your shell
comm -23 <(grep "REQUIRED" glue/ENVIRONMENT_VARIABLES.md | grep -E "^\|[A-Z_]+\|" | cut -d\| -f2 | sort) <(env | cut -d= -f1 | sort)
```

## Getting Help

1. **Check Documentation**: `glue/ENVIRONMENT_VARIABLES.md` - complete registry
2. **Check Examples**: `glue/lib/startup-validation.ts` - code examples
3. **Check ADR**: `glue/ADR-001-ENVIRONMENT-VALIDATION.md` - architecture rationale
4. **Check Schema**: `.env.schema` - all variables with defaults

## Checklist

Before deploying to production:

- [ ] All required variables set
- [ ] Strong secrets generated (64+ characters)
- [ ] `.env` in `.gitignore`
- [ ] No defaults from `.env.schema` in production `.env`
- [ ] All URLs using service names (not localhost) for Docker
- [ ] Ports don't conflict (check with `netstat -tulpn`)
- [ ] Log level set to `INFO` or `WARNING`
- [ ] `DEBUG=false` in production
- [ ] Secrets stored in secret manager (not in files)
- [ ] Test deployment validates successfully

# Security Fix Documentation - LeanAide ping.sh

## Issue Identified

**File:** `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\LeanAide\ping.sh`

**Security Vulnerability:** Hardcoded IP address (34.100.184.111)

**Current Content:**
```bash
curl -L http://34.100.184.111:5000
```

## Risk Assessment

1. **Exposure of Internal Infrastructure:** The hardcoded IP address reveals a specific server that may be:
   - A development/staging environment
   - A production server (security risk)
   - An internal service that should not be publicly exposed

2. **Compliance Violation:** Violates **Federation Constitution - Section 1, Law 5 (Configuration Explicitness)**:
   - No "Magic Defaults"
   - Every configurable value must be injected via Environment Variables
   - Code must validate `process.env` at startup

3. **Operational Risk:**
   - Cannot change endpoint without modifying source code
   - Script fails if IP address changes
   - Cannot run against different environments (dev/staging/prod)

## Recommended Fix

Replace the entire file with the following secure version:

```bash
#!/bin/bash
################################################################################
# LeanAide API Ping Script
# SECURITY FIX: Removed hardcoded IP address (34.100.184.111)
# Now uses LEANAIDE_API_URL environment variable as per Federation Constitution
#
# Environment Variables:
#   LEANAIDE_API_URL - Target API URL (REQUIRED)
#   TIMEOUT_MS       - Request timeout in milliseconds (default: 5000)
#
# Exit Codes:
#   0 - Success
#   1 - Configuration error
#   2 - API unreachable
################################################################################

set -euo pipefail

# Default values
TIMEOUT_MS=${TIMEOUT_MS:-5000}
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# SECURITY FIX: Fail fast if URL is not provided (no magic defaults)
if [[ -z "${LEANAIDE_API_URL:-}" ]]; then
    echo "ERROR: LEANAIDE_API_URL environment variable is required" >&2
    echo "Usage: LEANAIDE_API_URL=http://localhost:5000 ./ping.sh" >&2
    exit 1
fi

# Remove trailing slash if present
API_URL="${LEANAIDE_API_URL%/}"

echo "Pinging LeanAide API at: $API_URL"
echo "Timeout: ${TIMEOUT_MS}ms"
echo "---"

# Execute curl with timeout
if curl -s -o /dev/null -w "HTTP Status: %{http_code}\nTime: %{time_total}s\n" \
    --max-time "$TIMEOUT_SEC" \
    --connect-timeout "$TIMEOUT_SEC" \
    -L \
    "$API_URL"; then
    echo "---"
    echo "SUCCESS: API is reachable"
    exit 0
else
    echo "---"
    echo "ERROR: API is unreachable"
    exit 2
fi
```

## Migration Steps

1. **Backup the original file:**
   ```bash
   cp LeanAide/ping.sh LeanAide/ping.sh.backup
   ```

2. **Make file writable:**
   ```bash
   chmod 644 LeanAide/ping.sh
   ```

3. **Replace with secure version**

4. **Make executable:**
   ```bash
   chmod 755 LeanAide/ping.sh
   ```

5. **Update usage:**
   ```bash
   # Old (INSECURE):
   ./LeanAide/ping.sh

   # New (SECURE):
   LEANAIDE_API_URL=http://localhost:5000 ./LeanAide/ping.sh
   ```

## Environment Configuration

Add to `.env` or environment configuration:

```bash
# For local development
LEANAIDE_API_URL=http://localhost:5000

# For remote server
LEANAIDE_API_URL=http://leanaide-server:5000

# For production (with proper domain)
LEANAIDE_API_URL=https://leanaide.your-domain.com
```

## Compliance Achieved

After this fix, the script complies with:

- ✅ **Federation Constitution Law 5 (Configuration Explicitness)**
- ✅ **Federation Constitution Law 2 (Runtime Truth)** - fails fast if config missing
- ✅ **Mandatory timeouts** (Law of Networking & Discovery)
- ✅ **Proper exit codes**
- ✅ **Idempotent** (safe to run multiple times)

## Notes

- The original file is marked as read-only (`-r--r--r--`) which may be intentional for git tracking
- Consider whether this file should be:
  - Removed entirely (replaced by `glue/adapters/leanaide-adapter/probes/check_api.sh`)
  - Updated with the security fix
  - Kept as-is for backward compatibility (not recommended)

The new probe scripts in `glue/adapters/leanaide-adapter/probes/` are secure replacements that follow Federation Constitution requirements.

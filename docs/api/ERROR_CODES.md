# Error Codes Reference

Complete reference for all error codes in BubbleLab.

**Table of Contents:**
- [Overview](#overview)
- [HTTP Status Codes](#http-status-codes)
- [BubbleLab-Specific Errors](#bubblelab-specific-errors)
- [Service Bubble Errors](#service-bubble-errors)
- [Tool Bubble Errors](#tool-bubble-errors)
- [Validation Errors](#validation-errors)
- [Authentication Errors](#authentication-errors)
- [Rate Limit Errors](#rate-limit-errors)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Error Response Format](#error-response-format)

---

## Overview

BubbleLab uses standardized error codes for consistent error handling across all components.

### Error Categories

- **HTTP Errors**: Standard HTTP status codes
- **Application Errors**: BubbleLab-specific errors
- **Validation Errors**: Input validation failures
- **Authentication Errors**: Authentication and authorization failures
- **Rate Limit Errors**: Rate limiting and quota errors
- **Service Errors**: External service integration errors
- **System Errors**: Internal system errors

---

## HTTP Status Codes

### Success Codes

| Code | Name | Description |
|------|------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 202 | Accepted | Request accepted for processing |
| 204 | No Content | Request succeeded with no content |

### Redirection Codes

| Code | Name | Description |
|------|------|-------------|
| 301 | Moved Permanently | Resource permanently moved |
| 302 | Found | Resource temporarily moved |
| 304 | Not Modified | Resource not modified (conditional request) |

### Client Error Codes

| Code | Name | Description |
|------|------|-------------|
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required or failed |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (duplicate, etc.) |
| 422 | Unprocessable Entity | Semantic errors in request |
| 429 | Too Many Requests | Rate limit exceeded |

### Server Error Codes

| Code | Name | Description |
|------|------|-------------|
| 500 | Internal Server Error | Unexpected server error |
| 502 | Bad Gateway | Invalid response from upstream server |
| 503 | Service Unavailable | Service temporarily unavailable |
| 504 | Gateway Timeout | Upstream server timeout |

---

## BubbleLab-Specific Errors

### General Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `BL-1000` | 500 | Internal server error | Yes |
| `BL-1001` | 503 | Service maintenance | Yes |
| `BL-1002` | 500 | Database connection failed | Yes |
| `BL-1003` | 500 | Cache service unavailable | Yes |
| `BL-1004` | 503 | Rate limit exceeded (service-level) | Yes |
| `BL-1005` | 500 | Message queue error | Yes |
| `BL-1006` | 500 | Configuration error | No |
| `BL-1007` | 500 | Dependency service unavailable | Yes |
| `BL-1008` | 400 | Invalid request format | No |
| `BL-1009` | 500 | Timeout during processing | Yes |

**Example Response:**

```json
{
  "success": false,
  "error": {
    "code": "BL-1000",
    "message": "Internal server error",
    "details": "An unexpected error occurred while processing the request",
    "documentationUrl": "https://docs.bubblelab.io/errors/BL-1000"
  },
  "correlationId": "req_abc123",
  "timestamp": "2024-01-18T10:30:00Z"
}
```

---

### Flow Execution Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `BL-2000` | 404 | Flow not found | No |
| `BL-2001` | 400 | Invalid flow definition | No |
| `BL-2002` | 400 | Flow execution failed | No |
| `BL-2003` | 400 | Missing required parameter | No |
| `BL-2004` | 400 | Invalid parameter value | No |
| `BL-2005` | 400 | Circular dependency detected | No |
| `BL-2006` | 403 | Flow execution not permitted | No |
| `BL-2007` | 422 | Flow validation failed | No |
| `BL-2008` | 429 | Too many concurrent executions | Yes |
| `BL-2009` | 500 | Flow execution timeout | No |
| `BL-2010` | 400 | Bubble not found | No |
| `BL-2011` | 400 | Bubble execution failed | No |
| `BL-2012` | 400 | Invalid bubble configuration | No |
| `BL-2013` | 500 | Flow execution interrupted | Maybe |

**Example Response:**

```json
{
  "success": false,
  "error": {
    "code": "BL-2004",
    "message": "Missing required parameter",
    "details": {
      "parameter": "apiKey",
      "bubble": "openai-gpt",
      "step": 3
    },
    "documentationUrl": "https://docs.bubblelab.io/errors/BL-2004"
  },
  "correlationId": "req_def456",
  "timestamp": "2024-01-18T10:30:00Z"
}
```

---

### Credential Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `BL-3000` | 404 | Credential not found | No |
| `BL-3001` | 403 | Access to credential denied | No |
| `BL-3002` | 400 | Invalid credential format | No |
| `BL-3003` | 401 | Credential expired | No |
| `BL-3004` | 400 | Credential validation failed | No |
| `BL-3005` | 403 | Credential encryption failed | No |
| `BL-3006` | 429 | Too many credential access attempts | Yes |
| `BL-3007` | 400 | Duplicate credential name | No |
| `BL-3008` | 403 | Credential rotation in progress | No |

**Example Response:**

```json
{
  "success": false,
  "error": {
    "code": "BL-3003",
    "message": "Credential expired",
    "details": {
      "credentialId": "cred_abc123",
      "expiredAt": "2024-01-01T00:00:00Z",
      "rotateUrl": "https://api.bubblelab.io/v1/credentials/cred_abc123/rotate"
    },
    "documentationUrl": "https://docs.bubblelab.io/errors/BL-3003"
  },
  "correlationId": "req_ghi789",
  "timestamp": "2024-01-18T10:30:00Z"
}
```

---

## Service Bubble Errors

### HTTP Bubble Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `HTTP-100` | 400 | Invalid URL | No |
| `HTTP-101` | 400 | Invalid HTTP method | No |
| `HTTP-102` | 400 | Invalid headers format | No |
| `HTTP-103` | 403 | URL blocked by security policy | No |
| `HTTP-104` | 400 | Invalid request body | No |
| `HTTP-105` | 408 | Request timeout | Yes |
| `HTTP-106` | 500 | Network error | Yes |
| `HTTP-107` | 429 | Rate limit exceeded | Yes |
| `HTTP-108` | 500 | Response too large | No |
| `HTTP-109` | 500 | Connection failed | Yes |
| `HTTP-110` | 500 | SSL/TLS error | Yes |
| `HTTP-111` | 500 | DNS resolution failed | Yes |

**SSRF-Specific Errors:**

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `HTTP-200` | 403 | Private IP address blocked | No |
| `HTTP-201` | 403 | Localhost blocked | No |
| `HTTP-202` | 403 | Metadata endpoint blocked | No |
| `HTTP-203` | 403 | Invalid protocol | No |
| `HTTP-204` | 403 | Internal hostname blocked | No |

---

### AI Agent Bubble Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `AI-100` | 400 | Invalid model | No |
| `AI-101` | 400 | Invalid prompt | No |
| `AI-102` | 400 | Invalid temperature | No |
| `AI-103` | 400 | Invalid max tokens | No |
| `AI-104` | 401 | Invalid API key | No |
| `AI-105` | 429 | Provider rate limit exceeded | Yes |
| `AI-106` | 500 | Provider API error | Yes |
| `AI-107` | 408 | Generation timeout | No |
| `AI-108` | 400 | Content filtered | No |
| `AI-109` | 413 | Prompt too long | No |
| `AI-110` | 500 | Provider unavailable | Yes |

**Provider-Specific Errors:**

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `AI-200` | 401 | OpenAI authentication failed | No |
| `AI-201` | 429 | OpenAI rate limit exceeded | Yes |
| `AI-202` | 500 | OpenAI API error | Yes |
| `AI-203` | 401 | Anthropic authentication failed | No |
| `AI-204` | 429 | Anthropic rate limit exceeded | Yes |
| `AI-205` | 500 | Anthropic API error | Yes |

---

### PostgreSQL Bubble Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `PG-100` | 400 | Invalid query | No |
| `PG-101` | 400 | Invalid parameters | No |
| `PG-102` | 401 | Authentication failed | No |
| `PG-103` | 403 | Permission denied | No |
| `PG-104` | 404 | Table not found | No |
| `PG-105` | 409 | Unique constraint violation | No |
| `PG-106` | 408 | Query timeout | Maybe |
| `PG-107` | 500 | Connection pool exhausted | Yes |
| `PG-108` | 500 | Database connection failed | Yes |
| `PG-109` | 400 | Syntax error | No |
| `PG-110` | 500 | Transaction deadlock | Yes |
| `PG-111` | 429 | Too many connections | Yes |

---

### Slack Bubble Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `SLACK-100` | 400 | Invalid channel | No |
| `SLACK-101` | 400 | Invalid message format | No |
| `SLACK-102` | 400 | Invalid blocks format | No |
| `SLACK-103` | 404 | Channel not found | No |
| `SLACK-104` | 403 | Bot not in channel | No |
| `SLACK-105` | 429 | Rate limit exceeded | Yes |
| `SLACK-106` | 401 | Invalid bot token | No |
| `SLACK-107` | 500 | Slack API error | Yes |
| `SLACK-108` | 400 | Message too long | No |
| `SLACK-109` | 400 | Invalid thread timestamp | No |

---

### Storage Bubble Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `STORE-100` | 400 | Invalid operation | No |
| `STORE-101` | 400 | Invalid path | No |
| `STORE-102` | 404 | File not found | No |
| `STORE-103` | 409 | File already exists | No |
| `STORE-104` | 413 | File too large | No |
| `STORE-105` | 401 | Authentication failed | No |
| `STORE-106` | 403 | Access denied | No |
| `STORE-107` | 500 | Upload failed | Yes |
| `STORE-108` | 500 | Download failed | Yes |
| `STORE-109` | 500 | Storage service error | Yes |
| `STORE-110` | 400 | Invalid content type | No |

**Provider-Specific Errors:**

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `STORE-200` | 401 | AWS S3 authentication failed | No |
| `STORE-201` | 403 | AWS S3 access denied | No |
| `STORE-202` | 500 | AWS S3 service error | Yes |
| `STORE-203` | 401 | GCP authentication failed | No |
| `STORE-204` | 403 | GCP access denied | No |
| `STORE-205` | 500 | GCP service error | Yes |

---

## Tool Bubble Errors

### Code Edit Tool Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `CODE-100` | 400 | Invalid code | No |
| `CODE-101` | 400 | Unsupported language | No |
| `CODE-102` | 400 | Invalid operation | No |
| `CODE-103` | 422 | Parse error | No |
| `CODE-104` | 500 | Linter error | No |
| `CODE-105` | 500 | Formatter error | No |
| `CODE-106` | 400 | Invalid refactoring pattern | No |

---

### Chart.js Tool Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `CHART-100` | 400 | Invalid chart type | No |
| `CHART-101` | 400 | Invalid data format | No |
| `CHART-102` | 400 | Invalid options | No |
| `CHART-103` | 500 | Chart generation failed | No |
| `CHART-104` | 413 | Data too large | No |
| `CHART-105` | 400 | Invalid dimensions | No |

---

### Research Agent Tool Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `RESEARCH-100` | 400 | Invalid query | No |
| `RESEARCH-101` | 400 | Invalid research depth | No |
| `RESEARCH-102` | 404 | No results found | No |
| `RESEARCH-103` | 500 | Research service error | Yes |
| `RESEARCH-104` | 429 | Rate limit exceeded | Yes |
| `RESEARCH-105` | 400 | Invalid operation | No |

---

### Social Media Tool Errors

**Instagram:**

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `INSTA-100` | 400 | Invalid username | No |
| `INSTA-101` | 404 | Profile not found | No |
| `INSTA-102` | 429 | Rate limit exceeded | Yes |
| `INSTA-103` | 403 | Private profile | No |
| `INSTA-104` | 500 | Extraction failed | Yes |

**LinkedIn:**

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `LINKED-100` | 400 | Invalid profile URL | No |
| `LINKED-101` | 404 | Profile not found | No |
| `LINKED-102` | 429 | Rate limit exceeded | Yes |
| `LINKED-103` | 500 | Extraction failed | Yes |

**Twitter:**

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `TWITTER-100` | 400 | Invalid username | No |
| `TWITTER-101` | 404 | User not found | No |
| `TWITTER-102` | 429 | Rate limit exceeded | Yes |
| `TWITTER-103` | 500 | Extraction failed | Yes |

**YouTube:**

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `YOUTUBE-100` | 400 | Invalid video ID | No |
| `YOUTUBE-101` | 404 | Video not found | No |
| `YOUTUBE-102` | 429 | Rate limit exceeded | Yes |
| `YOUTUBE-103` | 500 | Extraction failed | Yes |

---

## Validation Errors

### Common Validation Errors

| Error Code | HTTP Status | Description | Example |
|------------|-------------|-------------|---------|
| `VAL-100` | 400 | Required field missing | Field 'name' is required |
| `VAL-101` | 400 | Invalid type | Field 'age' must be a number |
| `VAL-102` | 400 | Invalid format | Field 'email' must be valid email |
| `VAL-103` | 400 | Value out of range | Field 'age' must be between 0 and 150 |
| `VAL-104` | 400 | Invalid enum value | Field 'status' must be one of: active, inactive |
| `VAL-105` | 400 | String too short | Field 'name' must be at least 3 characters |
| `VAL-106` | 400 | String too long | Field 'name' must be at most 100 characters |
| `VAL-107` | 400 | Invalid regex pattern | Field 'phone' must match pattern |
| `VAL-108` | 400 | Invalid date format | Field 'date' must be ISO-8601 format |
| `VAL-109` | 400 | Invalid JSON | Request body must be valid JSON |
| `VAL-110` | 400 | Duplicate value | Field 'email' must be unique |

**Example Response:**

```json
{
  "success": false,
  "error": {
    "code": "VAL-100",
    "message": "Required field missing",
    "details": {
      "field": "name",
      "constraints": {
        "required": true
      }
    },
    "documentationUrl": "https://docs.bubblelab.io/errors/VAL-100"
  },
  "correlationId": "req_jkl012",
  "timestamp": "2024-01-18T10:30:00Z"
}
```

---

## Authentication Errors

| Error Code | HTTP Status | Description | Retry |
|------------|-------------|-------------|-------|
| `AUTH-100` | 401 | Invalid credentials | No |
| `AUTH-101` | 401 | Missing authentication | No |
| `AUTH-102` | 401 | Invalid token | No |
| `AUTH-103` | 401 | Token expired | No |
| `AUTH-104` | 403 | Insufficient permissions | No |
| `AUTH-105` | 403 | Invalid scope | No |
| `AUTH-106` | 401 | Invalid API key | No |
| `AUTH-107` | 401 | API key expired | No |
| `AUTH-108` | 403 | API key revoked | No |
| `AUTH-109` | 401 | Invalid signature | No |
| `AUTH-110` | 403 | MFA required | No |
| `AUTH-111` | 401 | Invalid MFA code | No |
| `AUTH-112` | 429 | Too many auth attempts | Yes |
| `AUTH-113` | 401 | Session expired | No |
| `AUTH-114` | 403 | Account locked | No |
| `AUTH-115` | 401 | Invalid OAuth state | No |

**Example Response:**

```json
{
  "success": false,
  "error": {
    "code": "AUTH-103",
    "message": "Token expired",
    "details": {
      "expiredAt": "2024-01-18T10:00:00Z",
      "refreshUrl": "https://auth.bubblelab.io/oauth/token"
    },
    "documentationUrl": "https://docs.bubblelab.io/errors/AUTH-103"
  },
  "correlationId": "req_mno345",
  "timestamp": "2024-01-18T10:30:00Z"
}
```

---

## Rate Limit Errors

### Rate Limit Error Codes

| Error Code | HTTP Status | Description | Retry After |
|------------|-------------|-------------|-------------|
| `RATE-100` | 429 | Rate limit exceeded | Varies |
| `RATE-101` | 429 | Burst rate exceeded | Varies |
| `RATE-102` | 429 | Daily quota exceeded | Varies |
| `RATE-103` | 429 | Concurrent limit exceeded | Varies |
| `RATE-104` | 429 | API rate limit exceeded | Varies |
| `RATE-105` | 429 | Storage quota exceeded | Varies |
| `RATE-106` | 429 | Webhook rate limit exceeded | Varies |

**Rate Limit Headers:**

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642579200
Retry-After: 60
```

**Example Response:**

```json
{
  "success": false,
  "error": {
    "code": "RATE-100",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "remaining": 0,
      "resetAt": "2024-01-18T11:00:00Z",
      "retryAfter": 60
    },
    "documentationUrl": "https://docs.bubblelab.io/errors/RATE-100"
  },
  "correlationId": "req_pqr678",
  "timestamp": "2024-01-18T10:30:00Z"
}
```

---

## Troubleshooting Guide

### Common Error Patterns

#### 1. Authentication Errors

**Symptoms:**
- 401 Unauthorized responses
- Token expired messages
- Invalid API key errors

**Solutions:**
1. Verify credentials are correct
2. Check token expiration
3. Refresh access tokens
4. Ensure API key is not revoked
5. Verify token has required scopes

---

#### 2. Rate Limiting Errors

**Symptoms:**
- 429 Too Many Requests
- Rate limit exceeded messages
- Requests being throttled

**Solutions:**
1. Implement exponential backoff
2. Reduce request frequency
3. Use batch operations
4. Upgrade plan for higher limits
5. Cache responses when possible

---

#### 3. Validation Errors

**Symptoms:**
- 400 Bad Request
- Missing required field
- Invalid format errors

**Solutions:**
1. Check request format
2. Validate data before sending
3. Ensure all required fields are present
4. Verify data types match schema
5. Check for character limits

---

#### 4. Timeout Errors

**Symptoms:**
- 408 Request Timeout
- Operation taking too long

**Solutions:**
1. Increase timeout value
2. Optimize query/operation
3. Break into smaller operations
4. Use async processing
5. Check for resource bottlenecks

---

### Error Handling Best Practices

#### 1. Always Check Success Field

```typescript
const result = await bubble.execute(params);

if (!result.success) {
  // Handle error
  console.error('Error:', result.error);
  return;
}

// Use result
console.log('Data:', result.data);
```

---

#### 2. Use Correlation IDs

```typescript
const result = await bubble.execute(params);

// Log correlation ID for debugging
console.log('Correlation ID:', result.correlationId);

// Include in error reports
if (!result.success) {
  reportError({
    error: result.error,
    correlationId: result.correlationId
  });
}
```

---

#### 3. Implement Retry Logic

```typescript
async function executeWithRetry(bubble, params, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const result = await bubble.execute(params);

      if (result.success) {
        return result;
      }

      // Check if error is retryable
      if (isRetryableError(result.error)) {
        const backoff = Math.pow(2, i) * 1000;
        await sleep(backoff);
        continue;
      }

      // Non-retryable error
      return result;
    } catch (error) {
      if (i === maxRetries - 1) {
        throw error;
      }

      const backoff = Math.pow(2, i) * 1000;
      await sleep(backoff);
    }
  }
}

function isRetryableError(error) {
  const retryableCodes = [
    'BL-1002', // Database connection failed
    'BL-1003', // Cache service unavailable
    'HTTP-105', // Request timeout
    'HTTP-106', // Network error
    'RATE-100'  // Rate limit exceeded
  ];

  return retryableCodes.includes(error.code);
}
```

---

#### 4. Graceful Degradation

```typescript
async function executeWithFallback(primary, fallback, params) {
  try {
    const result = await primary.execute(params);

    if (result.success) {
      return result;
    }

    // Check if we should use fallback
    if (shouldUseFallback(result.error)) {
      console.warn('Primary failed, using fallback:', result.error);
      return await fallback.execute(params);
    }

    return result;
  } catch (error) {
    console.error('Primary error, trying fallback:', error);
    return await fallback.execute(params);
  }
}

function shouldUseFallback(error) {
  const fallbackCodes = [
    'BL-1002', // Database connection failed
    'BL-1003', // Cache service unavailable
    'HTTP-109'  // Connection failed
  ];

  return fallbackCodes.includes(error.code);
}
```

---

## Error Response Format

### Standard Error Response

All errors follow this format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR-CODE",
    "message": "Human-readable error message",
    "details": {
      // Additional error-specific details
    },
    "documentationUrl": "https://docs.bubblelab.io/errors/ERROR-CODE"
  },
  "correlationId": "unique-request-id",
  "timestamp": "ISO-8601-timestamp"
}
```

---

### Error Details

Error details vary by error type but may include:

**Validation Errors:**
```json
{
  "field": "fieldName",
  "value": "providedValue",
  "constraints": {
    "required": true,
    "minLength": 3,
    "maxLength": 100
  }
}
```

**Rate Limit Errors:**
```json
{
  "limit": 100,
  "remaining": 0,
  "resetAt": "2024-01-18T11:00:00Z",
  "retryAfter": 60
}
```

**Authentication Errors:**
```json
{
  "expiredAt": "2024-01-18T10:00:00Z",
  "refreshUrl": "https://auth.bubblelab.io/oauth/token"
}
```

---

**Last Updated:** 2026-01-18
**Version:** 1.0.0
**Maintained By:** BubbleLab Core Team

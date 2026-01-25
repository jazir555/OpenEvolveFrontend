# Service Bubbles API Documentation

Complete API reference for all Service Bubbles in BubbleLab.

**Table of Contents:**
- [Overview](#overview)
- [HTTP Bubble](#http-bubble)
- [AI Agent Bubble](#ai-agent-bubble)
- [PostgreSQL Bubble](#postgresql-bubble)
- [Slack Bubble](#slack-bubble)
- [Storage Bubble](#storage-bubble)
- [Airtable Bubble](#airtable-bubble)
- [Apify Bubbles](#apify-bubbles)
- [Error Handling](#error-handling)
- [Security](#security)

---

## Overview

Service Bubbles are integration points that enable BubbleLab workflows to interact with external services. Each Service Bubble provides a standardized interface with common patterns for:

- **Authentication**: Credential management
- **Execution**: Service operation execution
- **Error Handling**: Standardized error responses
- **Validation**: Input validation and sanitization
- **Rate Limiting**: Request throttling
- **Logging**: Structured logging with correlation IDs

### Common Features

All Service Bubbles support:

- **Correlation IDs**: For request tracking
- **Timeout Configuration**: Per-request timeouts
- **Retry Logic**: Automatic retry with exponential backoff
- **Circuit Breakers**: Fail-fast for degraded services
- **Structured Logging**: JSON-formatted logs
- **Error Categorization**: Transient vs permanent errors

---

## HTTP Bubble

**Purpose**: Make HTTP requests to external APIs with built-in security and error handling.

### API Operations

#### `execute(context: BubbleContext)`

Executes an HTTP request to the specified URL.

**Parameters:**

```typescript
{
  url: string;              // Target URL (validated for SSRF)
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'HEAD' | 'OPTIONS';
  headers?: Record<string, string>;  // Request headers
  body?: any;               // Request body (JSON stringified)
  timeout?: number;         // Request timeout in milliseconds (default: 30000)
  followRedirects?: boolean; // Follow HTTP redirects (default: true)
  maxRedirects?: number;    // Maximum redirects to follow (default: 5)
  responseType?: 'json' | 'text' | 'buffer'; // Response type (default: 'json')
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    status: number;         // HTTP status code
    statusText: string;     // HTTP status text
    headers: Record<string, string>; // Response headers
    body: any;              // Response body (parsed based on responseType)
    responseTime: number;   // Response time in milliseconds
    size: number;           // Response size in bytes
  };
  error?: string;           // Error message if failed
  correlationId: string;    // Request correlation ID
}
```

**Example:**

```typescript
const httpBubble = new HttpBubble();

// GET request
const result = await httpBubble.execute({
  url: 'https://api.example.com/users/123',
  method: 'GET',
  headers: {
    'Authorization': 'Bearer token123',
    'Accept': 'application/json'
  },
  timeout: 10000
});

// POST request with body
const postResult = await httpBubble.execute({
  url: 'https://api.example.com/users',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: {
    name: 'John Doe',
    email: 'john@example.com'
  },
  timeout: 15000
});
```

**Error Responses:**

- `400`: Invalid parameters (URL, method, headers)
- `403`: URL blocked by security policy (SSRF protection)
- `408`: Request timeout
- `500`: Network error or service unavailable

**Security Features:**

- **SSRF Protection**: Blocks requests to internal/private IPs
- **Protocol Validation**: Only allows http/https
- **Metadata Blocking**: Prevents cloud metadata access
- **Hostname Filtering**: Blocks localhost and internal hostnames
- **URL Validation**: Strict URL format validation

**Rate Limits:**

- 100 requests per minute per bubble flow
- Burst limit: 10 requests per second
- Retry with exponential backoff: 1s, 2s, 4s, 8s, 16s

---

## AI Agent Bubble

**Purpose**: Generate text using Large Language Models (LLMs) with support for multiple providers.

### API Operations

#### `execute(context: BubbleContext)`

Generates text completion using configured AI model.

**Parameters:**

```typescript
{
  model: string;           // Model identifier (e.g., 'gpt-4', 'claude-3-opus')
  prompt: string;          // Input prompt
  systemPrompt?: string;   // System-level instructions
  temperature?: number;    // Sampling temperature (0.0-2.0, default: 0.7)
  maxTokens?: number;      // Maximum tokens to generate (default: 1000)
  topP?: number;          // Nucleus sampling (0.0-1.0, default: 1.0)
  topK?: number;          // Top-k sampling (default: 0)
  stop?: string[];        // Stop sequences
  timeout?: number;       // Request timeout in milliseconds (default: 60000)
  provider: 'openai' | 'anthropic' | 'cohere' | 'custom'; // AI provider
  apiKey?: string;        // API key (overrides stored credentials)
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    text: string;          // Generated text
    model: string;         // Model used
    usage: {
      promptTokens: number;     // Tokens in prompt
      completionTokens: number; // Tokens in completion
      totalTokens: number;      // Total tokens
    };
    finishReason: 'stop' | 'length' | 'content_filter'; // Why generation stopped
    responseTime: number;  // Generation time in milliseconds
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const aiBubble = new AiAgentBubble();

const result = await aiBubble.execute({
  model: 'gpt-4',
  prompt: 'Write a summary of the following article: {{article}}',
  systemPrompt: 'You are a helpful assistant that summarizes articles.',
  temperature: 0.7,
  maxTokens: 500,
  provider: 'openai',
  timeout: 30000
});

if (result.success) {
  console.log('Generated summary:', result.data.text);
  console.log('Tokens used:', result.data.usage.totalTokens);
}
```

**Supported Models:**

**OpenAI:**
- `gpt-4` - Most capable model
- `gpt-4-turbo` - Faster, cheaper GPT-4
- `gpt-3.5-turbo` - Fast, cost-effective
- `gpt-4-32k` - 32k context window

**Anthropic:**
- `claude-3-opus` - Most capable
- `claude-3-sonnet` - Balanced performance
- `claude-3-haiku` - Fastest, most cost-effective

**Cohere:**
- `command` - Text generation
- `command-light` - Faster, lighter

**Error Responses:**

- `400`: Invalid parameters
- `401`: Invalid API key
- `429`: Rate limit exceeded
- `500`: Provider API error

**Rate Limits:**

- Provider-dependent rate limits
- 100 requests per minute per flow
- Automatic retry with exponential backoff

**Best Practices:**

1. **Use appropriate temperature**: Lower (0.0-0.3) for factual, higher (0.7-1.0) for creative
2. **Set max tokens appropriately**: Avoid unnecessary token usage
3. **Provide system prompts**: Guide model behavior
4. **Handle content filters**: Respect finishReason
5. **Monitor usage**: Track token consumption

---

## PostgreSQL Bubble

**Purpose**: Execute SQL queries against PostgreSQL databases with connection pooling and prepared statements.

### API Operations

#### `execute(context: BubbleContext)`

Executes a SQL query and returns results.

**Parameters:**

```typescript
{
  query: string;           // SQL query with parameter placeholders ($1, $2, ...)
  params?: any[];          // Query parameters (prevents SQL injection)
  timeout?: number;        // Query timeout in milliseconds (default: 30000)
  connectionTimeout?: number; // Connection timeout (default: 10000)
  maxRows?: number;        // Maximum rows to return (default: 1000)
}

// Credentials (stored securely):
{
  host: string;            // Database host
  port: number;            // Database port (default: 5432)
  database: string;        // Database name
  user: string;            // Database user
  password: string;        // Database password
  ssl?: boolean;           // Use SSL (default: true)
  connectionTimeout?: number; // Connection timeout
  statementTimeout?: number;  // Statement timeout
  maxConnections?: number;    // Max pool size (default: 10)
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    rows: any[];           // Query result rows
    rowCount: number;      // Number of rows returned/affected
    fields: {
      name: string;        // Column name
      type: string;        // Column type
    }[];
    executionTime: number; // Query execution time in milliseconds
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const postgresBubble = new PostgreSQLBubble();

// SELECT query
const result = await postgresBubble.execute({
  query: 'SELECT * FROM users WHERE status = $1 AND created_at > $2',
  params: ['active', '2024-01-01'],
  maxRows: 100
});

// INSERT query
const insertResult = await postgresBubble.execute({
  query: 'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id',
  params: ['John Doe', 'john@example.com']
});

// UPDATE query
const updateResult = await postgresBubble.execute({
  query: 'UPDATE users SET status = $1 WHERE id = $2',
  params: ['active', 123]
});
```

**Security Features:**

- **Prepared Statements**: Prevents SQL injection
- **Parameterized Queries**: All values are escaped
- **Connection Pooling**: Efficient connection management
- **SSL/TLS**: Encrypted database connections
- **Row Limits**: Prevents large result sets
- **Query Timeouts**: Prevents long-running queries

**Rate Limits:**

- 50 queries per minute per flow
- Max 1000 rows per query
- Query timeout: 30 seconds
- Connection timeout: 10 seconds

**Best Practices:**

1. **Always use parameters**: Never concatenate values into queries
2. **Limit result sets**: Use maxRows and LIMIT clauses
3. **Use indexes**: Ensure columns in WHERE clauses are indexed
4. **Set appropriate timeouts**: Avoid blocking indefinitely
5. **Monitor performance**: Track slow queries

---

## Slack Bubble

**Purpose**: Send messages to Slack channels and users via Slack Web API.

### API Operations

#### `execute(context: BubbleContext)`

Sends a message to Slack.

**Parameters:**

```typescript
{
  channel: string;         // Channel ID or name (e.g., '#general', 'U12345')
  text?: string;           // Plain text message (use blocks for rich formatting)
  blocks?: any[];          // Block Kit blocks for rich formatting
  attachments?: any[];     // Legacy attachments (use blocks instead)
  threadTs?: string;       // Thread timestamp for threaded messages
  replyBroadcast?: boolean; // Reply in thread broadcast to channel
  username?: string;       // Override bot username
  iconUrl?: string;        // Override bot icon URL
  iconEmoji?: string;      // Override bot icon emoji
  unfurlLinks?: boolean;   // Enable automatic link unfurling
  unfurlMedia?: boolean;   // Enable automatic media unfurling
  timeout?: number;        // Request timeout in milliseconds (default: 10000)
}

// Credentials:
{
  webhookUrl?: string;     // Incoming webhook URL
  botToken?: string;       // Bot user OAuth token
  signingSecret?: string;  // Signing secret for verification
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    ok: boolean;
    channel: string;       // Channel ID
    ts: string;           // Message timestamp
    message: {
      botId: string;
      type: string;
      text: string;
      // ... other message fields
    };
    responseTime: number; // Response time in milliseconds
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const slackBubble = new SlackBubble();

// Simple text message
const result = await slackBubble.execute({
  channel: '#general',
  text: 'Hello from BubbleLab! 👋'
});

// Rich formatting with blocks
const richResult = await slackBubble.execute({
  channel: '#alerts',
  blocks: [
    {
      type: 'section',
      text: {
        type: 'mrkdwn',
        text: '*Alert:* Critical error detected in production'
      }
    },
    {
      type: 'divider'
    },
    {
      type: 'section',
      fields: [
        {
          type: 'mrkdwn',
          text: '*Service:*\nAPI Gateway'
        },
        {
          type: 'mrkdwn',
          text: '*Severity:*\n:rotating_light: Critical'
        }
      ]
    }
  ]
});

// Threaded reply
const threadResult = await slackBubble.execute({
  channel: '#general',
  text: 'This is a reply in a thread',
  threadTs: '1234567890.123456'
});
```

**Error Responses:**

- `400`: Invalid message format
- `403`: Missing permissions
- `404`: Channel not found
- `429`: Rate limit exceeded
- `500`: Slack API error

**Rate Limits:**

- Slack Workspace tier: Up to 1 message per second
- Webhook rate limit: Up to 1 message per second
- Automatic retry with exponential backoff

**Best Practices:**

1. **Use blocks**: Prefer blocks over attachments for rich formatting
2. **Keep messages concise**: Slack has message length limits
3. **Use threads**: Organize related messages in threads
4. **Handle rate limits**: Respect Slack's rate limits
5. **Test in development**: Use test channels before production

---

## Storage Bubble

**Purpose**: Store and retrieve files in cloud storage (S3, GCS, Azure Blob).

### API Operations

#### `execute(context: BubbleContext)`

Performs storage operations.

**Parameters (Upload):**

```typescript
{
  operation: 'upload';
  path: string;            // Storage path (e.g., 'files/document.pdf')
  content: string | Buffer; // File content
  contentType?: string;    // MIME type (e.g., 'application/pdf')
  metadata?: Record<string, string>; // Custom metadata
  timeout?: number;        // Upload timeout in milliseconds (default: 60000)
}

// Credentials (AWS S3):
{
  accessKeyId: string;
  secretAccessKey: string;
  region: string;          // e.g., 'us-east-1'
  bucket: string;          // Bucket name
  endpoint?: string;       // Custom endpoint (for S3-compatible services)
}

// Credentials (Google Cloud Storage):
{
  keyFile: string;        // Path to service account key JSON
  bucket: string;
}

// Credentials (Azure Blob):
{
  connectionString: string;
  container: string;
}
```

**Parameters (Download):**

```typescript
{
  operation: 'download';
  path: string;            // Storage path
  timeout?: number;        // Download timeout in milliseconds (default: 60000)
}
```

**Parameters (Delete):**

```typescript
{
  operation: 'delete';
  path: string;            // Storage path
  timeout?: number;        // Delete timeout in milliseconds (default: 10000)
}
```

**Parameters (List):**

```typescript
{
  operation: 'list';
  prefix?: string;         // Path prefix to filter
  maxResults?: number;     // Maximum results to return (default: 1000)
  timeout?: number;        // List timeout in milliseconds (default: 10000)
}
```

**Response:**

```typescript
// Upload/Download/Delete:
{
  success: boolean;
  data: {
    path: string;          // Storage path
    size?: number;         // File size in bytes
    contentType?: string;  // MIME type
    lastModified?: string; // ISO-8601 timestamp
    url?: string;          // Presigned URL (if requested)
    operationTime: number; // Operation time in milliseconds
  };
  error?: string;
  correlationId: string;
}

// List:
{
  success: boolean;
  data: {
    files: {
      path: string;
      size: number;
      contentType?: string;
      lastModified: string;
    }[];
    count: number;         // Number of files
    isTruncated: boolean;  // More results available
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const storageBubble = new StorageBubble();

// Upload file
const uploadResult = await storageBubble.execute({
  operation: 'upload',
  path: 'documents/report.pdf',
  content: fileBuffer,
  contentType: 'application/pdf',
  metadata: {
    'uploaded-by': 'bubblelab',
    'document-type': 'report'
  }
});

// Download file
const downloadResult = await storageBubble.execute({
  operation: 'download',
  path: 'documents/report.pdf'
});

// List files
const listResult = await storageBubble.execute({
  operation: 'list',
  prefix: 'documents/',
  maxResults: 100
});

// Delete file
const deleteResult = await storageBubble.execute({
  operation: 'delete',
  path: 'documents/old-report.pdf'
});
```

**Error Responses:**

- `400`: Invalid operation or parameters
- `403`: Permission denied
- `404`: File not found
- `413`: File too large (>5GB)
- `500`: Storage service error

**Rate Limits:**

- Provider-dependent (typically thousands per second)
- Multipart upload for files >100MB
- Automatic retry with exponential backoff

**Best Practices:**

1. **Use appropriate content types**: Ensure correct MIME types
2. **Set metadata**: Store useful metadata with files
3. **Use prefixes**: Organize files with path prefixes
4. **Handle large files**: Use multipart upload for >100MB
5. **Presigned URLs**: Generate temporary URLs for secure access

---

## Airtable Bubble

**Purpose**: Interact with Airtable bases, tables, and records.

### API Operations

#### `execute(context: BubbleContext)`

Performs Airtable operations.

**Parameters (Create):**

```typescript
{
  operation: 'create';
  table: string;           // Table name or ID
  fields: Record<string, any>; // Record fields
  typecast?: boolean;      // Automatic type conversion (default: false)
  timeout?: number;        // Request timeout in milliseconds (default: 10000)
}

// Credentials:
{
  apiKey: string;          // Airtable API key / personal access token
  baseId: string;          // Base ID
}
```

**Parameters (Read):**

```typescript
{
  operation: 'read';
  table: string;           // Table name or ID
  recordId?: string;       // Record ID (for single record)
  filterByFormula?: string; // Airtable formula for filtering
  sort?: Array<{          // Sorting
    field: string;
    direction: 'asc' | 'desc';
  }>;
  maxRecords?: number;     // Maximum records to return (default: 100)
  fields?: string[];       // Specific fields to return
  timeout?: number;        // Request timeout in milliseconds (default: 10000)
}
```

**Parameters (Update):**

```typescript
{
  operation: 'update';
  table: string;
  recordId: string;        // Record ID to update
  fields: Record<string, any>;
  typecast?: boolean;
  timeout?: number;
}
```

**Parameters (Delete):**

```typescript
{
  operation: 'delete';
  table: string;
  recordId: string;        // Record ID to delete
  timeout?: number;
}
```

**Parameters (List):**

```typescript
{
  operation: 'list';
  table: string;
  filterByFormula?: string;
  sort?: Array<{ field: string; direction: 'asc' | 'desc' }>;
  maxRecords?: number;
  fields?: string[];
  offset?: string;         // Pagination offset
  pageSize?: number;       // Page size (default: 100)
  timeout?: number;
}
```

**Response:**

```typescript
// Create/Update/Delete:
{
  success: boolean;
  data: {
    id: string;            // Record ID
    createdTime: string;   // ISO-8601 timestamp
    fields: Record<string, any>; // Record fields
  };
  error?: string;
  correlationId: string;
}

// Read (single record):
{
  success: boolean;
  data: {
    id: string;
    createdTime: string;
    fields: Record<string, any>;
  };
  error?: string;
  correlationId: string;
}

// List:
{
  success: boolean;
  data: {
    records: Array<{
      id: string;
      createdTime: string;
      fields: Record<string, any>;
    }>;
    offset?: string;       // Next page offset
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const airtableBubble = new AirtableBubble();

// Create record
const createResult = await airtableBubble.execute({
  operation: 'create',
  table: 'Contacts',
  fields: {
    'Name': 'John Doe',
    'Email': 'john@example.com',
    'Phone': '+1234567890'
  }
});

// Read single record
const readResult = await airtableBubble.execute({
  operation: 'read',
  table: 'Contacts',
  recordId: 'rec123456'
});

// List with filter
const listResult = await airtableBubble.execute({
  operation: 'list',
  table: 'Contacts',
  filterByFormula: '{Status} = "Active"',
  sort: [{ field: 'Name', direction: 'asc' }],
  maxRecords: 50
});

// Update record
const updateResult = await airtableBubble.execute({
  operation: 'update',
  table: 'Contacts',
  recordId: 'rec123456',
  fields: {
    'Status': 'Inactive'
  }
});

// Delete record
const deleteResult = await airtableBubble.execute({
  operation: 'delete',
  table: 'Contacts',
  recordId: 'rec123456'
});
```

**Error Responses:**

- `400`: Invalid request or parameters
- `401`: Invalid API key
- `403`: Insufficient permissions
- `404`: Table or record not found
- `413`: Request entity too large
- `422`: Unprocessable entity (validation error)
- `429`: Rate limit exceeded
- `500`: Airtable API error

**Rate Limits:**

- 5 requests per second per base
- Automatic retry with exponential backoff
- Offset-based pagination for large result sets

**Best Practices:**

1. **Use specific fields**: Only request fields you need
2. **Filter with formulas**: Use filterByFormula for efficient filtering
3. **Batch operations**: Use bulk endpoints when available
4. **Handle pagination**: Use offset for large result sets
5. **Type conversion**: Enable typecast for automatic type handling

---

## Apify Bubbles

**Purpose**: Run Apify actors for web scraping and automation.

### Available Actors

- **Google Maps Scraper**: `google-maps-scraper`
- **Instagram Hashtag Scraper**: `instagram-hashtag-scraper`
- **Instagram Scraper**: `instagram-scraper`
- **LinkedIn Jobs Scraper**: `linkedin-jobs-scraper`
- **LinkedIn Posts Search**: `linkedin-posts-search`
- **LinkedIn Profile Posts**: `linkedin-profile-posts`
- **TikTok Scraper**: `tiktok-scraper`
- **Twitter Scraper**: `twitter-scraper`
- **YouTube Scraper**: `youtube-scraper`

### API Operations

#### `execute(context: BubbleContext)`

Runs an Apify actor with specified input.

**Parameters:**

```typescript
{
  actorId: string;         // Actor identifier (e.g., 'instagram-scraper')
  input: any;              // Actor-specific input object
  timeout?: number;        // Run timeout in milliseconds (default: 300000)
  memory?: number;         // Memory in MB (default: 1024)
  waitForFinish?: boolean; // Wait for completion (default: true)
}

// Credentials:
{
  apiToken: string;        // Apify API token
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    id: string;            // Run ID
    actId: string;         // Actor ID
    status: 'READY' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'TIMED-OUT' | 'ABORTED';
    startedAt: string;     // ISO-8601 timestamp
    finishedAt?: string;   // ISO-8601 timestamp
    datasetId?: string;    // Dataset ID with results
    results?: any[];       // Dataset items (if waitForFinish)
    usage: {
      actorBuildId: string;
      duration: number;    // Run duration in seconds
      memSize: number;     // Memory used in MB
      cpuSeconds: number;  // CPU time used
    };
  };
  error?: string;
  correlationId: string;
}
```

**Example (Instagram Scraper):**

```typescript
const apifyBubble = new ApifyBubble();

const result = await apifyBubble.execute({
  actorId: 'instagram-scraper',
  input: {
    usernames: ['instagram'],
    resultsLimit: 10,
    addParentData: false
  },
  timeout: 300000
});

if (result.success) {
  console.log('Scraped profiles:', result.data.results);
}
```

**Example (Google Maps Scraper):**

```typescript
const mapsResult = await apifyBubble.execute({
  actorId: 'google-maps-scraper',
  input: {
    searchStrings: ['restaurant in New York'],
    maxCrawledPlaces: 20,
    language: 'en'
  }
});
```

**Error Responses:**

- `400`: Invalid input or actor ID
- `401`: Invalid API token
- `404`: Actor not found
- `429`: Rate limit exceeded or insufficient credits
- `500`: Apify platform error

**Rate Limits:**

- Limited by Apify account credits
- Actor-specific rate limits
- Automatic retry for transient errors

**Best Practices:**

1. **Set reasonable timeouts**: Scraping can take time
2. **Limit results**: Use resultsLimit to control costs
2. **Handle large datasets**: Results may be in dataset, not inline
3. **Monitor usage**: Track credits and execution time
4. **Validate input**: Each actor has specific input requirements

---

## Error Handling

All Service Bubbles use standardized error handling.

### Error Response Format

```typescript
{
  success: false;
  error: string;           // Human-readable error message
  errorCode: string;       // Machine-readable error code
  details?: any;           // Additional error details
  correlationId: string;   // Request correlation ID
  timestamp: string;       // ISO-8601 timestamp
}
```

### Error Categories

**Transient Errors (Retry Recommended):**
- `NETWORK_ERROR`: Network connectivity issues
- `TIMEOUT`: Request timeout
- `RATE_LIMITED`: Rate limit exceeded
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

**Permanent Errors (Do Not Retry):**
- `INVALID_PARAMETERS`: Invalid input parameters
- `UNAUTHORIZED`: Authentication failed
- `FORBIDDEN`: Permission denied
- `NOT_FOUND`: Resource not found
- `CONFLICT`: Resource conflict

**Validation Errors:**
- `VALIDATION_ERROR`: Input validation failed
- `SCHEMA_ERROR`: Schema mismatch
- `TYPE_ERROR`: Type mismatch

### Error Handling Best Practices

1. **Check success field first**: Always check `success` before accessing `data`
2. **Log correlation IDs**: Include correlation IDs in logs for debugging
3. **Categorize errors**: Use error codes to determine retry strategy
4. **Handle transient errors**: Retry with exponential backoff
5. **Fail fast for permanent errors**: Don't retry authentication errors

---

## Security

### Authentication

All Service Bubbles support secure credential management:

```typescript
// Credentials are stored securely and never logged
// Use CredentialType enum for type safety
const credentials = {
  apiKey: 'sk-...',
  // ... other credential fields
};
```

### Security Features

**HTTP Bubble:**
- SSRF protection
- URL validation
- Protocol filtering
- Private IP blocking
- Metadata endpoint blocking

**PostgreSQL Bubble:**
- Prepared statements (SQL injection prevention)
- SSL/TLS connections
- Parameterized queries
- Connection pool security

**AI Agent Bubble:**
- API key encryption
- Prompt injection protection
- Content filtering
- Rate limiting

**Storage Bubble:**
- Encrypted connections
- Presigned URLs with expiration
- Access control lists
- Metadata validation

**Airtable Bubble:**
- API key management
- Table-level permissions
- Field-level validation
- Rate limiting

**Apify Bubbles:**
- API token management
- Input validation
- Output sanitization

### Best Practices

1. **Never log credentials**: Credentials are automatically redacted
2. **Use environment variables**: Store credentials in environment
3. **Rotate credentials regularly**: Implement credential rotation
4. **Use least privilege**: Grant minimum required permissions
5. **Monitor access**: Audit credential usage
6. **Encrypt sensitive data**: Use encryption at rest and in transit

---

## Performance Optimization

### Connection Pooling

Services that support it use connection pooling:
- PostgreSQL: Max 10 connections per pool
- HTTP: Reuses connections with keep-alive
- Storage: Connection pooling for S3/GCS

### Caching

Consider caching for:
- Frequently accessed data
- Expensive operations
- Reference data

### Rate Limiting

All Service Bubbles implement:
- Per-flow rate limits
- Burst limits
- Automatic retry with backoff
- Circuit breaker pattern

### Timeouts

Set appropriate timeouts:
- HTTP: 30 seconds default
- Database: 30 seconds default
- AI Agent: 60 seconds default
- Storage: 60 seconds default

---

**Last Updated:** 2026-01-18
**Version:** 1.0.0
**Maintained By:** BubbleLab Core Team

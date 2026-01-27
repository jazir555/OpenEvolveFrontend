# BubbleLab Integration Guide

Complete guide for integrating BubbleLab with external systems and services.

**Table of Contents:**
- [Overview](#overview)
- [Integration Patterns](#integration-patterns)
- [Authentication](#authentication)
- [Webhooks](#webhooks)
- [API Integration](#api-integration)
- [Event-Driven Integration](#event-driven-integration)
- [Database Integration](#database-integration)
- [Third-Party Services](#third-party-services)
- [Custom Bubbles](#custom-bubbles)
- [Testing Integrations](#testing-integrations)
- [Monitoring & Debugging](#monitoring--debugging)
- [Security Best Practices](#security-best-practices)

---

## Overview

BubbleLab provides multiple integration patterns to connect with external systems:

- **Service Bubbles**: Pre-built integrations with popular services
- **API Integration**: REST/GraphQL API client
- **Webhooks**: Event-driven notifications
- **Database Integration**: Direct database connections
- **Custom Bubbles**: Custom integration logic

### Integration Capabilities

**Supported Protocols:**
- REST APIs
- GraphQL APIs
- Webhooks
- Database connections (PostgreSQL, MySQL, SQLite, MSSQL, Oracle)
- File protocols (SFTP, S3, GCS, Azure Blob)

**Authentication Methods:**
- API Keys
- OAuth 2.0
- JWT Tokens
- Basic Auth
- Custom Headers

**Data Formats:**
- JSON
- XML
- CSV
- Form-encoded
- Binary

---

## Integration Patterns

### 1. Synchronous Request/Response

**Use Case:** API calls that return immediate results

```typescript
const result = await httpBubble.execute({
  url: 'https://api.example.com/data',
  method: 'GET',
  headers: {
    'Authorization': 'Bearer token123'
  }
});
```

**Best For:**
- CRUD operations
- Data retrieval
- API queries
- Real-time processing

**Considerations:**
- Timeouts affect flow execution
- Errors require handling
- Rate limits apply

---

### 2. Asynchronous Event Processing

**Use Case:** Long-running operations or background tasks

```typescript
// Start async operation
const startResult = await httpBubble.execute({
  url: 'https://api.example.com/jobs',
  method: 'POST',
  body: { task: 'export-data' }
});

const jobId = startResult.data.id;

// Poll for completion
let status = 'running';
while (status === 'running') {
  await sleep(5000);
  const checkResult = await httpBubble.execute({
    url: `https://api.example.com/jobs/${jobId}`,
    method: 'GET'
  });
  status = checkResult.data.status;
}
```

**Best For:**
- Long-running operations
- Batch processing
- Data exports
- Report generation

**Considerations:**
- Requires polling mechanism
- Implement timeout limits
- Handle job failures

---

### 3. Webhook Reception

**Use Case:** Receive notifications from external systems

```typescript
// Set up webhook endpoint in BubbleLab API
// External system sends POST request to endpoint

// In webhook handler:
const webhookData = request.body;

// Process webhook data
const result = await processData(webhookData);

// Respond with 200 OK
return { success: true };
```

**Best For:**
- Event notifications
- Status updates
- Real-time triggers
- Third-party callbacks

**Considerations:**
- Must respond quickly (<5 seconds)
- Implement retry handling
- Verify webhook signatures
- Handle duplicate events

---

### 4. Batch Processing

**Use Case:** Process multiple items efficiently

```typescript
const items = [...]; // Large dataset

// Process in batches
const batchSize = 100;
for (let i = 0; i < items.length; i += batchSize) {
  const batch = items.slice(i, i + batchSize);

  // Parallel processing within batch
  const results = await Promise.all(
    batch.map(item => processItem(item))
  );

  // Batch insert
  await postgresBubble.execute({
    query: 'INSERT INTO results (data) VALUES ($1)',
    params: [results]
  });
}
```

**Best For:**
- Bulk data processing
- ETL operations
- Data migration
- Batch updates

**Considerations:**
- Memory usage
- Rate limits
- Error handling per item
- Progress tracking

---

## Authentication

### API Key Authentication

**Setup:**

```typescript
// Store API key in credentials
const credentials = {
  apiKey: 'your-api-key'
};

// Use in bubble
const result = await httpBubble.execute({
  url: 'https://api.example.com/data',
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${credentials.apiKey}`,
    'X-API-Key': credentials.apiKey
  }
});
```

**Best Practices:**
- Store keys securely (environment variables)
- Rotate keys regularly
- Use minimum required permissions
- Monitor key usage

---

### OAuth 2.0

**Setup:**

```typescript
// OAuth 2.0 flow
// 1. Redirect user to authorization URL
const authUrl = `https://api.example.com/oauth/authorize?` +
  `client_id=${clientId}&` +
  `redirect_uri=${redirectUri}&` +
  `response_type=code&` +
  `scope=read,write`;

// 2. User authorizes, receives code
// 3. Exchange code for access token
const tokenResult = await httpBubble.execute({
  url: 'https://api.example.com/oauth/token',
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

const accessToken = tokenResult.data.access_token;

// 4. Use access token
const result = await httpBubble.execute({
  url: 'https://api.example.com/data',
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});
```

**Best Practices:**
- Use PKCE for public clients
- Store tokens securely
- Implement token refresh
- Handle token expiration

---

### JWT Tokens

**Setup:**

```typescript
// Generate JWT token
const jwt = require('jsonwebtoken');

const token = jwt.sign(
  {
    user_id: '123',
    role: 'admin'
  },
  secretKey,
  {
    expiresIn: '1h',
    issuer: 'bubblelab',
    audience: 'api.example.com'
  }
);

// Use in API request
const result = await httpBubble.execute({
  url: 'https://api.example.com/data',
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

**Best Practices:**
- Use strong signing keys
- Set appropriate expiration
- Include claims needed
- Validate tokens server-side

---

### Basic Authentication

**Setup:**

```typescript
// Basic auth (use HTTPS only!)
const credentials = {
  username: 'user',
  password: 'pass'
};

const auth = Buffer.from(
  `${credentials.username}:${credentials.password}`
).toString('base64');

const result = await httpBubble.execute({
  url: 'https://api.example.com/data',
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
- Consider API keys instead

---

## Webhooks

### Receiving Webhooks

**Setup in BubbleLab API:**

```typescript
// Express.js route handler
app.post('/webhooks/:provider', async (req, res) => {
  const { provider } = req.params;
  const payload = req.body;

  // Verify webhook signature
  const signature = req.headers['x-webhook-signature'];
  if (!verifySignature(payload, signature, provider)) {
    return res.status(401).json({ error: 'Invalid signature' });
  }

  // Process webhook
  try {
    await processWebhook(provider, payload);
    res.status(200).json({ success: true });
  } catch (error) {
    console.error('Webhook processing failed:', error);
    res.status(500).json({ error: 'Processing failed' });
  }
});
```

**Signature Verification:**

```typescript
function verifySignature(payload, signature, secret) {
  const hmac = crypto.createHmac('sha256', secret);
  const digest = hmac.update(JSON.stringify(payload)).digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(digest)
  );
}
```

**Handling Duplicate Events:**

```typescript
const processedEvents = new Set();

async function processWebhook(provider, payload) {
  const eventId = payload.id;

  // Check for duplicates
  if (processedEvents.has(eventId)) {
    console.log('Duplicate event, skipping:', eventId);
    return;
  }

  // Process event
  await handleEvent(payload);

  // Mark as processed
  processedEvents.add(eventId);

  // Cleanup old events
  if (processedEvents.size > 10000) {
    const oldEvents = Array.from(processedEvents).slice(0, 5000);
    oldEvents.forEach(id => processedEvents.delete(id));
  }
}
```

---

### Sending Webhooks

**Setup:**

```typescript
// Send webhook notification
async function sendWebhook(url, payload, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const result = await httpBubble.execute({
        url: url,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Webhook-Source': 'bubblelab'
        },
        body: payload,
        timeout: 10000
      });

      if (result.success) {
        console.log('Webhook sent successfully');
        return result;
      }
    } catch (error) {
      console.error(`Webhook attempt ${i + 1} failed:`, error);

      if (i === retries - 1) {
        throw error;
      }

      // Exponential backoff
      await sleep(Math.pow(2, i) * 1000);
    }
  }
}
```

**Best Practices:**
- Implement retries with backoff
- Use idempotent operations
- Include event metadata
- Handle failures gracefully
- Log all webhook attempts

---

## API Integration

### REST API Integration

**Basic Example:**

```typescript
// GET request
const getResult = await httpBubble.execute({
  url: 'https://api.example.com/users/123',
  method: 'GET',
  headers: {
    'Authorization': 'Bearer token123',
    'Accept': 'application/json'
  }
});

// POST request
const postResult = await httpBubble.execute({
  url: 'https://api.example.com/users',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: {
    name: 'John Doe',
    email: 'john@example.com'
  }
});

// PUT request
const putResult = await httpBubble.execute({
  url: 'https://api.example.com/users/123',
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json'
  },
  body: {
    name: 'Jane Doe'
  }
});

// DELETE request
const deleteResult = await httpBubble.execute({
  url: 'https://api.example.com/users/123',
  method: 'DELETE'
});
```

**Pagination:**

```typescript
// Paginated results
let page = 1;
const allResults = [];

while (true) {
  const result = await httpBubble.execute({
    url: `https://api.example.com/users?page=${page}&per_page=100`,
    method: 'GET'
  });

  if (!result.success || result.data.length === 0) {
    break;
  }

  allResults.push(...result.data);
  page++;
}

console.log(`Total results: ${allResults.length}`);
```

**Rate Limiting:**

```typescript
// Rate-limited requests
const rateLimit = {
  requests: 100,
  per: 60000, // 100 requests per minute
  tokens: 100,
  lastRefill: Date.now()
};

async function makeRateLimitedRequest(url, options) {
  // Refill tokens
  const now = Date.now();
  const elapsed = now - rateLimit.lastRefill;
  const tokensToAdd = Math.floor(elapsed / (rateLimit.per / rateLimit.requests));

  rateLimit.tokens = Math.min(
    rateLimit.requests,
    rateLimit.tokens + tokensToAdd
  );
  rateLimit.lastRefill = now;

  // Wait for tokens
  while (rateLimit.tokens < 1) {
    await sleep(100);
    // Recalculate tokens
    // ... (same as above)
  }

  // Consume token
  rateLimit.tokens--;

  // Make request
  return await httpBubble.execute({ url, ...options });
}
```

---

### GraphQL API Integration

**Example:**

```typescript
// GraphQL query
const query = `
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      email
      posts {
        id
        title
        content
      }
    }
  }
`;

const result = await httpBubble.execute({
  url: 'https://api.example.com/graphql',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer token123'
  },
  body: {
    query: query,
    variables: {
      id: '123'
    }
  }
});

const user = result.data.data.user;
console.log('User:', user.name);
console.log('Posts:', user.posts.length);
```

**Mutation:**

```typescript
const mutation = `
  mutation CreateUser($input: CreateUserInput!) {
    createUser(input: $input) {
      id
      name
      email
    }
  }
`;

const result = await httpBubble.execute({
  url: 'https://api.example.com/graphql',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: {
    query: mutation,
    variables: {
      input: {
        name: 'John Doe',
        email: 'john@example.com'
      }
    }
  }
});

const user = result.data.data.createUser;
```

---

## Event-Driven Integration

### Pub/Sub Pattern

**Publisher:**

```typescript
// Publish event
async function publishEvent(topic, event) {
  await httpBubble.execute({
    url: `https://pubsub.example.com/topics/${topic}`,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: {
      event: event.type,
      data: event.data,
      timestamp: new Date().toISOString(),
      id: generateUniqueId()
    }
  });
}

// Usage
await publishEvent('user.created', {
  type: 'user.created',
  data: {
    userId: '123',
    name: 'John Doe'
  }
});
```

**Subscriber:**

```typescript
// Subscribe to events
async function subscribeToTopic(topic, handler) {
  // This would typically be a long-lived connection
  // For example, using Server-Sent Events or WebSocket

  while (true) {
    const result = await httpBubble.execute({
      url: `https://pubsub.example.com/topics/${topic}/pull`,
      method: 'POST',
      body: {
        maxMessages: 10
      }
    });

    if (result.success && result.data.messages) {
      for (const message of result.data.messages) {
        await handler(message);
      }
    }

    await sleep(1000); // Poll interval
  }
}

// Usage
subscribeToTopic('user.created', async (message) => {
  console.log('User created:', message.data);
  await processNewUser(message.data);
});
```

---

### Message Queues

**Producer:**

```typescript
// Send message to queue
async function sendMessage(queue, message) {
  await httpBubble.execute({
    url: `https://mq.example.com/queues/${queue}`,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: {
      body: JSON.stringify(message),
      attributes: {
        contentType: 'application/json',
        timestamp: new Date().toISOString()
      }
    }
  });
}

// Usage
await sendMessage('user-events', {
  type: 'user.created',
  userId: '123',
  timestamp: new Date().toISOString()
});
```

**Consumer:**

```typescript
// Receive message from queue
async function receiveMessage(queue) {
  const result = await httpBubble.execute({
    url: `https://mq.example.com/queues/${queue}/messages`,
    method: 'GET',
    params: {
      max: 1
    }
  });

  if (result.success && result.data.messages.length > 0) {
    const message = result.data.messages[0];

    // Process message
    await processMessage(message);

    // Delete message
    await httpBubble.execute({
      url: `https://mq.example.com/queues/${queue}/messages/${message.id}`,
      method: 'DELETE'
    });

    return message;
  }

  return null;
}

// Usage
while (true) {
  const message = await receiveMessage('user-events');
  if (message) {
    console.log('Processed message:', message.id);
  } else {
    await sleep(1000);
  }
}
```

---

## Database Integration

### PostgreSQL Integration

**Connection Setup:**

```typescript
const postgresBubble = new PostgreSQLBubble();

// Query database
const result = await postgresBubble.execute({
  query: 'SELECT * FROM users WHERE status = $1',
  params: ['active'],
  maxRows: 100
});

// Insert data
await postgresBubble.execute({
  query: 'INSERT INTO users (name, email, status) VALUES ($1, $2, $3)',
  params: ['John Doe', 'john@example.com', 'active']
});

// Update data
await postgresBubble.execute({
  query: 'UPDATE users SET status = $1 WHERE id = $2',
  params: ['inactive', 123]
});

// Delete data
await postgresBubble.execute({
  query: 'DELETE FROM users WHERE id = $1',
  params: [123]
});
```

**Transaction Management:**

```typescript
// Execute multiple queries in transaction
async function executeTransaction(queries) {
  // Start transaction
  await postgresBubble.execute({
    query: 'BEGIN'
  });

  try {
    // Execute queries
    for (const query of queries) {
      await postgresBubble.execute(query);
    }

    // Commit transaction
    await postgresBubble.execute({
      query: 'COMMIT'
    });
  } catch (error) {
    // Rollback on error
    await postgresBubble.execute({
      query: 'ROLLBACK'
    });
    throw error;
  }
}

// Usage
await executeTransaction([
  {
    query: 'INSERT INTO accounts (user_id, balance) VALUES ($1, $2)',
    params: [123, 100]
  },
  {
    query: 'INSERT INTO transactions (account_id, amount) VALUES ($1, $2)',
    params: [123, 100]
  }
]);
```

---

### Data Synchronization

**One-way Sync:**

```typescript
// Sync data from API to database
async function syncDataToDatabase(apiUrl, table, primaryKey) {
  // Fetch from API
  const apiResult = await httpBubble.execute({
    url: apiUrl,
    method: 'GET'
  });

  if (!apiResult.success) {
    throw new Error('API request failed');
  }

  // Upsert to database
  for (const item of apiResult.data) {
    await postgresBubble.execute({
      query: `
        INSERT INTO ${table} (data)
        VALUES ($1)
        ON CONFLICT (${primaryKey})
        DO UPDATE SET data = $1
      `,
      params: [JSON.stringify(item)]
    });
  }

  console.log(`Synced ${apiResult.data.length} items to ${table}`);
}

// Usage
await syncDataToDatabase(
  'https://api.example.com/users',
  'users',
  'id'
);
```

**Two-way Sync:**

```typescript
// Sync data bidirectionally
async function bidirectionalSync(apiUrl, table, primaryKey) {
  // Fetch from API
  const apiResult = await httpBubble.execute({
    url: apiUrl,
    method: 'GET'
  });

  // Fetch from database
  const dbResult = await postgresBubble.execute({
    query: `SELECT data FROM ${table}`
  });

  // Compare and merge
  const apiMap = new Map(apiResult.data.map(item => [item[primaryKey], item]));
  const dbMap = new Map(dbResult.data.rows.map(row => [row.data[primaryKey], row.data]));

  // Find items to update in API
  for (const [id, dbItem] of dbMap) {
    const apiItem = apiMap.get(id);
    if (!apiItem || isNewer(dbItem, apiItem)) {
      await httpBubble.execute({
        url: `${apiUrl}/${id}`,
        method: 'PUT',
        body: dbItem
      });
    }
  }

  // Find items to update in database
  for (const [id, apiItem] of apiMap) {
    const dbItem = dbMap.get(id);
    if (!dbItem || isNewer(apiItem, dbItem)) {
      await postgresBubble.execute({
        query: `
          INSERT INTO ${table} (data)
          VALUES ($1)
          ON CONFLICT (${primaryKey})
          DO UPDATE SET data = $1
        `,
        params: [JSON.stringify(apiItem)]
      });
    }
  }
}
```

---

## Third-Party Services

### Salesforce Integration

```typescript
// Salesforce REST API
const salesforceUrl = 'https://your-instance.salesforce.com';

// Authenticate
const authResult = await httpBubble.execute({
  url: `${salesforceUrl}/services/oauth2/token`,
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: new URLSearchParams({
    grant_type: 'password',
    client_id: clientId,
    client_secret: clientSecret,
    username: username,
    password: password
  })
});

const accessToken = authResult.data.access_token;

// Query records
const queryResult = await httpBubble.execute({
  url: `${salesforceUrl}/services/data/v56.0/query?q=SELECT+Id,+Name+FROM+Account`,
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});

// Create record
const createResult = await httpBubble.execute({
  url: `${salesforceUrl}/services/data/v56.0/sobjects/Account`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json'
  },
  body: {
    Name: 'New Account'
  }
});
```

---

### Shopify Integration

```typescript
// Shopify REST API
const shopifyUrl = 'https://your-store.myshopify.com';

// Get products
const productsResult = await httpBubble.execute({
  url: `${shopifyUrl}/admin/api/2023-10/products.json`,
  method: 'GET',
  headers: {
    'X-Shopify-Access-Token': accessToken
  }
});

// Create product
const createResult = await httpBubble.execute({
  url: `${shopifyUrl}/admin/api/2023-10/products.json`,
  method: 'POST',
  headers: {
    'X-Shopify-Access-Token': accessToken,
    'Content-Type': 'application/json'
  },
  body: {
    product: {
      title: 'New Product',
      body_html: '<p>Description</p>',
      vendor: 'Vendor',
      product_type: 'Type',
      variants: [{
        price: '10.00',
        sku: 'SKU123'
      }]
    }
  }
});
```

---

### Stripe Integration

```typescript
// Stripe API
const stripeUrl = 'https://api.stripe.com/v1';

// Create charge
const chargeResult = await httpBubble.execute({
  url: `${stripeUrl}/charges`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${stripeSecretKey}`,
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: new URLSearchParams({
    amount: '1000',
    currency: 'usd',
    source: 'tok_visa', // Test token
    description: 'Payment for order'
  })
});

// Create customer
const customerResult = await httpBubble.execute({
  url: `${stripeUrl}/customers`,
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${stripeSecretKey}`,
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: new URLSearchParams({
    'email': 'customer@example.com',
    'name': 'Customer Name',
    'description': 'New customer'
  })
});
```

---

## Custom Bubbles

### Creating a Custom Service Bubble

```typescript
import { ServiceBubble } from '@bubblelab/bubble-core';

class CustomServiceBubble extends ServiceBubble {
  name = 'custom-service';
  description = 'My custom service integration';

  async execute(context) {
    const { params } = context;

    // Validate input
    if (!params.url) {
      throw new Error('URL is required');
    }

    // Perform custom logic
    const result = await this.callCustomAPI(params.url);

    return {
      success: true,
      data: result,
      correlationId: context.correlationId
    };
  }

  async callCustomAPI(url) {
    // Custom API logic
    const response = await fetch(url);
    return await response.json();
  }
}

// Register the bubble
export default new CustomServiceBubble();
```

---

### Creating a Custom Tool Bubble

```typescript
import { ToolBubble } from '@bubblelab/bubble-core';

class CustomToolBubble extends ToolBubble {
  name = 'custom-tool';
  description = 'My custom data processing tool';

  async execute(context) {
    const { input } = context;

    // Validate input
    if (!input.data) {
      throw new Error('Data is required');
    }

    // Perform custom processing
    const result = this.processData(input.data);

    return {
      success: true,
      data: result,
      correlationId: context.correlationId
    };
  }

  processData(data) {
    // Custom data processing logic
    return data.map(item => ({
      ...item,
      processed: true,
      timestamp: new Date().toISOString()
    }));
  }
}

// Register the tool
export default new CustomToolBubble();
```

---

## Testing Integrations

### Unit Testing

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { HttpBubble } from '@bubblelab/bubble-core';

describe('HTTP Integration', () => {
  let httpBubble;

  beforeEach(() => {
    httpBubble = new HttpBubble();
  });

  it('should fetch data from API', async () => {
    const result = await httpBubble.execute({
      url: 'https://api.example.com/test',
      method: 'GET',
      headers: {
        'Authorization': 'Bearer test-token'
      }
    });

    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });

  it('should handle errors gracefully', async () => {
    const result = await httpBubble.execute({
      url: 'https://api.example.com/invalid',
      method: 'GET'
    });

    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();
  });
});
```

---

### Integration Testing

```typescript
describe('Integration Test', () => {
  it('should complete full workflow', async () => {
    // Step 1: Fetch data from API
    const apiResult = await httpBubble.execute({
      url: 'https://api.example.com/data',
      method: 'GET'
    });

    expect(apiResult.success).toBe(true);

    // Step 2: Process data
    const processedResult = await customTool.execute({
      input: { data: apiResult.data }
    });

    expect(processedResult.success).toBe(true);

    // Step 3: Store in database
    const dbResult = await postgresBubble.execute({
      query: 'INSERT INTO processed_data (data) VALUES ($1)',
      params: [JSON.stringify(processedResult.data)]
    });

    expect(dbResult.success).toBe(true);
  });
});
```

---

### Mocking External APIs

```typescript
import { vi } from 'vitest';

// Mock HTTP bubble
vi.mock('@bubblelab/bubble-core', () => ({
  HttpBubble: vi.fn().mockImplementation(() => ({
    execute: vi.fn().mockResolvedValue({
      success: true,
      data: { id: '123', name: 'Test' },
      correlationId: 'test-id'
    })
  }))
}));

// Test with mock
it('should handle API response', async () => {
  const httpBubble = new HttpBubble();
  const result = await httpBubble.execute({
    url: 'https://api.example.com/test',
    method: 'GET'
  });

  expect(result.data.name).toBe('Test');
});
```

---

## Monitoring & Debugging

### Logging

```typescript
// Structured logging
function logIntegrationEvent(event, data) {
  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    event: event,
    data: data,
    correlationId: data.correlationId
  }));
}

// Usage
logIntegrationEvent('api.request.started', {
  url: 'https://api.example.com/data',
  method: 'GET',
  correlationId: generateCorrelationId()
});
```

---

### Error Tracking

```typescript
// Capture and report errors
async function withErrorTracking(operation) {
  const correlationId = generateCorrelationId();

  try {
    const result = await operation();
    return result;
  } catch (error) {
    // Log error
    console.error(JSON.stringify({
      timestamp: new Date().toISOString(),
      error: error.message,
      stack: error.stack,
      correlationId: correlationId
    }));

    // Send to error tracking service
    await httpBubble.execute({
      url: 'https://error-tracking.example.com/api/errors',
      method: 'POST',
      body: {
        error: error.message,
        stack: error.stack,
        correlationId: correlationId,
        context: {
          operation: operation.name
        }
      }
    });

    throw error;
  }
}

// Usage
const result = await withErrorTracking(async () => {
  return await httpBubble.execute({
    url: 'https://api.example.com/data',
    method: 'GET'
  });
});
```

---

### Performance Monitoring

```typescript
// Track operation performance
async function trackPerformance(operation, name) {
  const startTime = Date.now();
  const startMemory = process.memoryUsage().heapUsed;

  try {
    const result = await operation();

    const duration = Date.now() - startTime;
    const memoryUsed = process.memoryUsage().heapUsed - startMemory;

    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      operation: name,
      duration: duration,
      memoryUsed: memoryUsed,
      success: true
    }));

    return result;
  } catch (error) {
    const duration = Date.now() - startTime;

    console.error(JSON.stringify({
      timestamp: new Date().toISOString(),
      operation: name,
      duration: duration,
      success: false,
      error: error.message
    }));

    throw error;
  }
}

// Usage
const result = await trackPerformance(
  () => httpBubble.execute({
    url: 'https://api.example.com/data',
    method: 'GET'
  }),
  'api.request'
);
```

---

## Security Best Practices

### Input Validation

```typescript
// Validate all inputs
function validateInput(input, schema) {
  const errors = [];

  for (const [field, rules] of Object.entries(schema)) {
    const value = input[field];

    if (rules.required && !value) {
      errors.push(`${field} is required`);
    }

    if (rules.type && typeof value !== rules.type) {
      errors.push(`${field} must be ${rules.type}`);
    }

    if (rules.pattern && !rules.pattern.test(value)) {
      errors.push(`${field} is invalid`);
    }
  }

  if (errors.length > 0) {
    throw new Error(errors.join(', '));
  }
}

// Usage
validateInput(input, {
  url: { required: true, type: 'string', pattern: /^https?:\/\// },
  method: { required: true, type: 'string', enum: ['GET', 'POST', 'PUT', 'DELETE'] },
  headers: { required: false, type: 'object' }
});
```

---

### Output Sanitization

```typescript
// Sanitize output data
function sanitizeOutput(data) {
  const sensitiveFields = ['password', 'token', 'secret', 'key'];

  if (typeof data === 'object' && data !== null) {
    const sanitized = { ...data };

    for (const field of sensitiveFields) {
      if (field in sanitized) {
        sanitized[field] = '***REDACTED***';
      }
    }

    return sanitized;
  }

  return data;
}

// Usage
const result = await httpBubble.execute({
  url: 'https://api.example.com/data',
  method: 'GET'
});

console.log('Sanitized result:', sanitizeOutput(result.data));
```

---

### Rate Limiting

```typescript
// Implement rate limiting
class RateLimiter {
  constructor(requests, perMilliseconds) {
    this.requests = requests;
    this.perMilliseconds = perMilliseconds;
    this.tokens = requests;
    this.lastRefill = Date.now();
  }

  async waitForToken() {
    while (this.tokens < 1) {
      const now = Date.now();
      const elapsed = now - this.lastRefill;
      const tokensToAdd = Math.floor(
        (elapsed / this.perMilliseconds) * this.requests
      );

      this.tokens = Math.min(this.requests, this.tokens + tokensToAdd);
      this.lastRefill = now;

      if (this.tokens < 1) {
        const waitTime = Math.ceil(
          ((1 - this.tokens) * this.perMilliseconds) / this.requests
        );
        await sleep(waitTime);
      }
    }

    this.tokens--;
  }
}

// Usage
const limiter = new RateLimiter(100, 60000); // 100 requests per minute

await limiter.waitForToken();
const result = await httpBubble.execute({
  url: 'https://api.example.com/data',
  method: 'GET'
});
```

---

**Last Updated:** 2026-01-18
**Version:** 1.0.0
**Maintained By:** BubbleLab Core Team

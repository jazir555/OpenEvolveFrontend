# Webhook Bubble Quick Reference Guide

## Basic Usage

### 1. Receive and Validate Webhook
```typescript
import { WebhookBubble } from '@bubblelab/bubble-core';

const webhook = new WebhookBubble({
  operation: 'receiveWebhook',
  path: '/webhooks/github',
  headers: {
    'x-github-event': 'push',
    'x-hub-signature-256': 'sha256=...',
    'content-type': 'application/json',
  },
  body: { ref: 'main', repository: { name: 'test' } },
  signature: 'sha256=...',
  secret: process.env.GITHUB_WEBHOOK_SECRET,
  signatureAlgorithm: 'hmac-sha256',
  timestamp: new Date().toISOString(),
  maxAge: 300000, // 5 minutes
  maxPayloadSize: 10485760, // 10MB
});

const result = await webhook.execute();
console.log(result.webhookId, result.validated);
```

### 2. Verify Signature
```typescript
const verify = new WebhookBubble({
  operation: 'verifySignature',
  payload: rawPayload,
  signature: headers['x-hub-signature-256'],
  secret: process.env.WEBHOOK_SECRET,
  algorithm: 'hmac-sha256',
  provider: 'github',
  timestamp: headers['x-github-delivery'],
});

const verification = await verify.execute();
if (!verification.result.valid) {
  throw new Error('Invalid signature');
}
```

### 3. Register Event Handler
```typescript
const register = new WebhookBubble({
  operation: 'registerHandler',
  eventType: 'push',
  handlerUrl: 'https://api.example.com/webhooks/handle',
  filter: { repository: 'my-repo' },
  timeout: 10000,
  retries: 3,
});

const handler = await register.execute();
console.log('Handler ID:', handler.result.handlerId);
```

### 4. Retry Failed Webhook
```typescript
const retry = new WebhookBubble({
  operation: 'retryFailedWebhook',
  webhookId: 'webhook-id',
  retryCount: 0,
  maxRetries: 5,
  backoffMs: 60000, // 1 minute
});

const retryResult = await retry.execute();
console.log('Status:', retryResult.result.status);
console.log('Next retry:', retryResult.result.nextRetryAt);
```

### 5. Get Retry Status
```typescript
const status = new WebhookBubble({
  operation: 'getRetryStatus',
  webhookId: 'webhook-id',
});

const retryStatus = await status.execute();
console.log('Retry count:', retryStatus.result.retryCount);
console.log('History:', retryStatus.result.retryHistory);
```

## Supported Providers

### GitHub
```typescript
{
  provider: 'github',
  headers: {
    'x-github-event': 'push',
    'x-github-delivery': '12345',
    'x-hub-signature-256': 'sha256=...',
  }
}
```

### Stripe
```typescript
{
  provider: 'stripe',
  headers: {
    'x-stripe-signature': 't=...,v1=...',
  }
}
```

### Slack
```typescript
{
  provider: 'slack',
  headers: {
    'x-slack-request-timestamp': '1234567890',
    'x-slack-signature': 'v0=...',
  }
}
```

### Twilio
```typescript
{
  provider: 'twilio',
  headers: {
    'x-twilio-signature': '...',
  }
}
```

## Security Configuration

### Environment Variables
```bash
# Required
WEBHOOK_SECRET=your-secret-key-here
WEBHOOK_SIGNING_KEY=your-signing-key

# Optional (with defaults)
WEBHOOK_MAX_PAYLOAD_SIZE=10485760  # 10MB
WEBHOOK_RATE_LIMIT_RECEIVE=100      # per minute
WEBHOOK_RATE_LIMIT_DISPATCH=50      # per minute
WEBHOOK_MAX_RETRIES=5
WEBHOOK_RETRY_BACKOFF_MS=60000      # 1 minute
WEBHOOK_TIMEOUT_MS=10000            # 10 seconds
```

### Signature Algorithms
- `hmac-sha1` - GitHub, GitLab
- `hmac-sha256` - Stripe, Slack, default
- `aws-v4` - AWS services

## Rate Limits

### Receive Operations
- **Limit:** 100 webhooks/minute per path
- **Window:** Sliding 60-second window
- **Error:** Includes reset time

### Dispatch Operations
- **Limit:** 50 dispatches/minute
- **Window:** Sliding 60-second window
- **Per Target:** Independent limits

## Retry Schedule

### Exponential Backoff
```
Attempt 1: Immediate (or 1 minute)
Attempt 2: 2 minutes (1 × 2)
Attempt 3: 4 minutes (2 × 2)
Attempt 4: 8 minutes (4 × 2)
Attempt 5: 16 minutes (8 × 2)
```

### Retry States
- `pending` - Waiting for retry
- `success` - Delivered successfully
- `failed` - Last attempt failed
- `exhausted` - All retries attempted

## Filtering and Search

### List Webhooks
```typescript
const list = new WebhookBubble({
  operation: 'listWebhooks',
  limit: 50,
  offset: 0,
  filter: {
    path: '/webhooks/github',
    provider: 'github',
    startDate: '2025-01-01T00:00:00Z',
    endDate: '2025-01-31T23:59:59Z',
  },
});

const webhooks = await list.execute();
```

### Get Statistics
```typescript
const stats = new WebhookBubble({
  operation: 'getStats',
  path: '/webhooks/stripe',
  timeRange: 'day', // hour, day, week, month
});

const metrics = await stats.execute();
console.log('Total received:', metrics.result.metrics.totalReceived);
console.log('Validation failure rate:', metrics.result.metrics.validationFailureRate);
```

## Error Handling

### Common Errors
```typescript
// Rate limit exceeded
if (result.error.includes('Rate limit exceeded')) {
  const resetTime = extractResetTime(result.error);
  await waitUntil(resetTime);
}

// Signature validation failed
if (result.error.includes('Signature validation failed')) {
  // Reject webhook
  return { status: 401, error: 'Invalid signature' };
}

// Payload too large
if (result.error.includes('exceeds maximum allowed size')) {
  return { status: 413, error: 'Payload too large' };
}

// Timestamp validation failed
if (result.error.includes('Timestamp validation failed')) {
  return { status: 400, error: 'Webhook too old' };
}
```

## Best Practices

### 1. Always Verify Signatures
```typescript
const webhook = new WebhookBubble({
  operation: 'receiveWebhook',
  // ... other params
  signature: headers['x-hub-signature'],
  secret: process.env.WEBHOOK_SECRET,
  signatureAlgorithm: 'hmac-sha256',
});

if (!webhook.result.validated) {
  // Reject unverified webhooks
}
```

### 2. Implement Rate Limiting
```typescript
// Use built-in rate limiting
// Configure appropriate limits for your use case
// Monitor rate limit breaches
```

### 3. Handle Retries Properly
```typescript
// Check retry status before processing
const status = await new WebhookBubble({
  operation: 'getRetryStatus',
  webhookId: id,
}).execute();

if (status.result.status === 'exhausted') {
  // Alert team, manual intervention needed
}
```

### 4. Monitor and Log
```typescript
console.log({
  webhookId: result.webhookId,
  provider: result.provider,
  eventType: result.eventType,
  validated: result.validated,
  processed: result.processed,
  timestamp: new Date().toISOString(),
});
```

### 5. Use Appropriate Timeouts
```typescript
const handler = await new WebhookBubble({
  operation: 'registerHandler',
  handlerUrl: url,
  timeout: 10000, // 10 seconds
  retries: 3,
}).execute();
```

## Integration Examples

### Express.js
```typescript
app.post('/webhooks/:provider', async (req, res) => {
  const webhook = new WebhookBubble({
    operation: 'receiveWebhook',
    path: `/webhooks/${req.params.provider}`,
    headers: req.headers,
    body: req.body,
    signature: req.headers['x-hub-signature'],
    secret: process.env[`${req.params.provider.toUpperCase()}_SECRET`],
  });

  const result = await webhook.execute();

  if (!result.success) {
    return res.status(400).json({ error: result.error });
  }

  res.status(200).json({ webhookId: result.webhookId });
});
```

### Fastify
```typescript
fastify.post('/webhooks/:provider', async (request, reply) => {
  const webhook = new WebhookBubble({
    operation: 'receiveWebhook',
    path: request.url,
    headers: request.headers,
    body: request.body,
    signature: request.headers['x-hub-signature'],
    secret: process.env.WEBHOOK_SECRET,
  });

  const result = await webhook.execute();

  if (!result.success) {
    return reply.status(400).send({ error: result.error });
  }

  return reply.send({ webhookId: result.webhookId });
});
```

## Testing

### Unit Test Example
```typescript
test('receiveWebhook validates signature', async () => {
  const payload = { test: 'data' };
  const signature = generateSignature(payload, 'secret');

  const webhook = new WebhookBubble({
    operation: 'receiveWebhook',
    path: '/test',
    headers: {},
    body: payload,
    signature,
    secret: 'secret',
    signatureAlgorithm: 'hmac-sha256',
  });

  const result = await webhook.execute();
  expect(result.validated).toBe(true);
});
```

## Troubleshooting

### Issue: Rate Limit Exceeded
**Solution:** Implement backoff and retry after reset time

### Issue: Signature Validation Fails
**Solution:** Verify secret key and algorithm match provider

### Issue: Webhook Too Old
**Solution:** Check system time synchronization

### Issue: Payload Too Large
**Solution:** Increase `maxPayloadSize` or reduce payload

### Issue: Handler Not Found
**Solution:** Verify handler is registered and active

## Performance Tips

1. **Use In-Memory Storage** for development
2. **Use Redis** for production (not yet implemented)
3. **Batch Operations** when possible
4. **Monitor Memory Usage** with high webhook volume
5. **Implement Cleanup** for old webhooks

## Migration Guide

### From Basic Webhook Handler
```typescript
// Before
app.post('/webhook', (req, res) => {
  console.log('Webhook received:', req.body);
  res.send('OK');
});

// After
app.post('/webhook', async (req, res) => {
  const webhook = new WebhookBubble({
    operation: 'receiveWebhook',
    path: '/webhook',
    headers: req.headers,
    body: req.body,
  });

  const result = await webhook.execute();
  res.send({ webhookId: result.webhookId });
});
```

## Additional Resources

- **Implementation Summary:** `WEBHOOK_BUBBLE_IMPLEMENTATION_SUMMARY.md`
- **Source Code:** `src/bubbles/service-bubble/webhook-bubble.ts`
- **Test Examples:** See test files in same directory
- **API Reference:** See JSDoc comments in source

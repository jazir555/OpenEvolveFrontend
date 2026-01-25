# Stripe Service Bubble - Quick Reference Guide

**Last Updated:** 2026-01-18
**Status:** Production Ready

---

## Quick Start

### 1. Setup Credentials
```typescript
// In your backend or environment
const stripeKey = 'sk_test_abc123...'; // Get from Stripe Dashboard

// Store in BubbleLab credential system
await createCredential({
  credentialType: 'STRIPE_CRED',
  value: stripeKey,
  name: 'Stripe Test Key',
  isDefault: true
});
```

### 2. Basic Usage
```typescript
import { StripeBubble } from '@bubblelab/bubble-core';

// Create a payment intent
const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000, // $10.00 in cents
  currency: 'usd',
  customer: 'cus_abc123'
}).execute();

console.log(payment.result.clientSecret); // Use with Stripe.js
```

---

## Operation Catalog

### Payment Operations

#### createPaymentIntent
```typescript
const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000,                    // Amount in cents ($10.00)
  currency: 'usd',                 // 3-letter currency code
  customer: 'cus_abc123',          // Optional: Customer ID
  paymentMethod: 'pm_abc123',      // Optional: Payment method ID
  description: 'Order #1234',      // Optional: Description
  metadata: { orderId: '1234' },   // Optional: Custom metadata
  confirm: false,                  // Optional: Confirm immediately
  captureMethod: 'automatic'       // automatic | manual
}).execute();
```

#### confirmPayment
```typescript
const confirmed = await new StripeBubble({
  operation: 'confirmPayment',
  paymentIntentId: 'pi_abc123',
  paymentMethod: 'pm_xyz789'      // Optional: Different payment method
}).execute();
```

#### refundPayment
```typescript
const refund = await new StripeBubble({
  operation: 'refundPayment',
  paymentIntentId: 'pi_abc123',
  amount: 500,                     // Optional: Partial refund (in cents)
  reason: 'requested_by_customer', // duplicate | fraudulent | requested_by_customer | other
  metadata: { reason: 'Customer request' }
}).execute();
```

### Customer Operations

#### createCustomer
```typescript
const customer = await new StripeBubble({
  operation: 'createCustomer',
  email: 'customer@example.com',
  name: 'John Doe',
  phone: '+1234567890',
  description: 'VIP Customer',
  metadata: { source: 'website' }
}).execute();
```

#### getCustomer
```typescript
const customer = await new StripeBubble({
  operation: 'getCustomer',
  customerId: 'cus_abc123'
}).execute();
```

#### updateCustomer
```typescript
const updated = await new StripeBubble({
  operation: 'updateCustomer',
  customerId: 'cus_abc123',
  email: 'newemail@example.com',
  name: 'Jane Doe',
  metadata: { updated: '2026-01-18' }
}).execute();
```

### Subscription Operations

#### createSubscription
```typescript
const subscription = await new StripeBubble({
  operation: 'createSubscription',
  customer: 'cus_abc123',
  priceId: 'price_abc123',
  quantity: 1,                     // Optional: Default 1
  trialPeriodDays: 14,             // Optional: Trial period
  paymentBehavior: 'default_incomplete', // default_incomplete | allow_incomplete | error_if_incomplete
  metadata: { plan: 'premium' }
}).execute();
```

#### cancelSubscription
```typescript
const canceled = await new StripeBubble({
  operation: 'cancelSubscription',
  subscriptionId: 'sub_abc123',
  cancelAtPeriodEnd: true          // Optional: Default true
}).execute();
```

#### updateSubscription
```typescript
const updated = await new StripeBubble({
  operation: 'updateSubscription',
  subscriptionId: 'sub_abc123',
  priceId: 'price_xyz789',         // Optional: New price
  quantity: 2,                     // Optional: New quantity
  prorationBehavior: 'create_prorations', // create_prorations | always_invoice | none
  metadata: { upgraded: 'true' }
}).execute();
```

### Invoice Operations

#### createInvoice
```typescript
const invoice = await new StripeBubble({
  operation: 'createInvoice',
  customer: 'cus_abc123',
  description: 'January Services',
  autoAdvance: true,               // Optional: Default true
  collectionMethod: 'charge_automatically', // charge_automatically | send_invoice
  metadata: { period: '2026-01' }
}).execute();
```

#### getInvoice
```typescript
const invoice = await new StripeBubble({
  operation: 'getInvoice',
  invoiceId: 'in_abc123'
}).execute();
```

#### listInvoices
```typescript
const list = await new StripeBubble({
  operation: 'listInvoices',
  customer: 'cus_abc123',          // Optional: Filter by customer
  limit: 10,                       // Optional: Default 10
  startingAfter: 'in_xyz789',      // Optional: Pagination cursor
  status: 'paid'                   // Optional: draft | open | paid | uncollectible | void
}).execute();
```

### Product & Price Operations

#### createProduct
```typescript
const product = await new StripeBubble({
  operation: 'createProduct',
  name: 'Premium Plan',
  description: 'Monthly subscription',
  images: ['https://example.com/image.png'],
  statementDescriptor: 'PREMIUM PLAN',
  unitLabel: 'seat',
  metadata: { category: 'subscription' }
}).execute();
```

#### createPrice
```typescript
const price = await new StripeBubble({
  operation: 'createPrice',
  product: 'prod_abc123',
  unitAmount: 2900,                // $29.00 in cents
  currency: 'usd',
  recurring: {
    interval: 'month',             // day | week | month | year
    intervalCount: 1,              // Optional: Default 1
    usageType: 'licensed'          // licensed | metered
  },
  nickname: 'Monthly Premium',
  metadata: { tier: 'premium' }
}).execute();
```

### Webhook Operations

#### handleWebhook
```typescript
const event = await new StripeBubble({
  operation: 'handleWebhook',
  payload: req.body,               // Raw webhook payload string
  signature: req.headers['stripe-signature'],
  secret: process.env.STRIPE_WEBHOOK_SECRET
}).execute();

console.log('Event type:', event.result.type);
console.log('Event data:', event.result.data);
```

---

## Response Format

All operations return:
```typescript
{
  operation: 'operationName',
  result: {
    // Operation-specific data
    id: string,
    success: boolean,
    error: string, // Empty string if success
    ...otherFields
  }
}
```

### Success Example
```typescript
{
  operation: 'createPaymentIntent',
  result: {
    id: 'pi_abc123',
    amount: 1000,
    currency: 'usd',
    status: 'requires_payment_method',
    clientSecret: 'pi_abc123_secret_xyz',
    description: 'Order #1234',
    createdAt: '2026-01-18T12:00:00.000Z',
    success: true,
    error: ''
  }
}
```

### Error Example
```typescript
{
  operation: 'createPaymentIntent',
  result: {
    id: '',
    amount: 1000,
    currency: 'usd',
    status: '',
    createdAt: '',
    success: false,
    error: 'Stripe API error: 400 - Invalid amount'
  }
}
```

---

## Error Handling

### Automatic Retries
Transient errors are automatically retried:
- Network errors (ECONNREFUSED, ETIMEDOUT)
- HTTP 503 (Service Unavailable)
- HTTP 502 (Bad Gateway)
- HTTP 429 (Rate Limit)

### Circuit Breaker
After 5 consecutive failures, the circuit breaker opens:
- Subsequent requests fail immediately
- Reopens after 60 seconds
- Requires 2 successes to close completely

### Manual Error Handling
```typescript
try {
  const result = await new StripeBubble({
    operation: 'createPaymentIntent',
    amount: 1000,
    currency: 'usd'
  }).execute();

  if (!result.result.success) {
    console.error('Payment failed:', result.result.error);
    // Handle error
  }
} catch (error) {
  console.error('Operation failed:', error);
  // Handle circuit breaker or other errors
}
```

---

## Best Practices

### 1. Always Use Customer Objects
```typescript
// Good
const customer = await new StripeBubble({
  operation: 'createCustomer',
  email: 'customer@example.com'
}).execute();

const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000,
  currency: 'usd',
  customer: customer.result.id
}).execute();

// Avoid (creates customer implicitly)
const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000,
  currency: 'usd'
  // No customer
}).execute();
```

### 2. Use Metadata for Tracking
```typescript
const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000,
  currency: 'usd',
  metadata: {
    orderId: 'order_123',
    userId: 'user_456',
    source: 'mobile_app',
    version: '2.1.0'
  }
}).execute();
```

### 3. Handle Webhooks Securely
```typescript
// Always verify webhook signatures
const event = await new StripeBubble({
  operation: 'handleWebhook',
  payload: req.body,
  signature: req.headers['stripe-signature'],
  secret: process.env.STRIPE_WEBHOOK_SECRET // Required!
}).execute();

if (!event.result.success) {
  console.error('Invalid webhook signature');
  return res.status(400).send('Invalid signature');
}

// Process event
switch (event.result.type) {
  case 'payment_intent.succeeded':
    // Handle successful payment
    break;
  case 'invoice.paid':
    // Handle paid invoice
    break;
}
```

### 4. Use Idempotency for Repeated Operations
```typescript
// Add idempotency key via metadata
const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000,
  currency: 'usd',
  metadata: {
    idempotencyKey: `order_${orderId}_payment_${timestamp}`
  }
}).execute();
```

---

## Common Workflows

### One-Time Payment
```typescript
// 1. Create customer
const customer = await new StripeBubble({
  operation: 'createCustomer',
  email: 'customer@example.com',
  name: 'John Doe'
}).execute();

// 2. Create payment intent
const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000,
  currency: 'usd',
  customer: customer.result.id,
  metadata: { orderId: 'order_123' }
}).execute();

// 3. Use clientSecret with Stripe.js on frontend
const { clientSecret } = payment.result;
```

### Subscription with Trial
```typescript
// 1. Create product
const product = await new StripeBubble({
  operation: 'createProduct',
  name: 'Premium Plan'
}).execute();

// 2. Create price
const price = await new StripeBubble({
  operation: 'createPrice',
  product: product.result.id,
  unitAmount: 2900,
  currency: 'usd',
  recurring: { interval: 'month' }
}).execute();

// 3. Create customer
const customer = await new StripeBubble({
  operation: 'createCustomer',
  email: 'customer@example.com'
}).execute();

// 4. Create subscription with trial
const subscription = await new StripeBubble({
  operation: 'createSubscription',
  customer: customer.result.id,
  priceId: price.result.id,
  trialPeriodDays: 14
}).execute();
```

### Invoice and Payment
```typescript
// 1. Create customer
const customer = await new StripeBubble({
  operation: 'createCustomer',
  email: 'customer@example.com'
}).execute();

// 2. Create invoice
const invoice = await new StripeBubble({
  operation: 'createInvoice',
  customer: customer.result.id,
  description: 'Consulting Services',
  autoAdvance: true,
  collectionMethod: 'send_invoice'
}).execute();

// 3. List invoices
const invoices = await new StripeBubble({
  operation: 'listInvoices',
  customer: customer.result.id,
  status: 'open'
}).execute();
```

### Refund Flow
```typescript
// 1. Create payment
const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000,
  currency: 'usd',
  confirm: true,
  paymentMethod: 'pm_abc123'
}).execute();

// 2. Process refund
const refund = await new StripeBubble({
  operation: 'refundPayment',
  paymentIntentId: payment.result.id,
  amount: 500, // Partial refund
  reason: 'requested_by_customer'
}).execute();
```

---

## Testing

### Test Mode
```typescript
// Use test keys (start with sk_test_)
const testKey = 'sk_test_abc123...';

// Create test payment intent
const payment = await new StripeBubble({
  operation: 'createPaymentIntent',
  amount: 1000,
  currency: 'usd',
  credentials: { [CredentialType.STRIPE_CRED]: testKey }
}).execute();
```

### Test Cards
Use these card numbers with Stripe test mode:
- **4242 4242 4242 4242** - Success
- **4000 0000 0000 0002** - Card declined
- **4000 0000 0000 9995** - Insufficient funds
- **4000 0025 0000 3155** - Requires authentication

---

## Environment Variables

```bash
# Required
STRIPE_API_KEY=sk_test_abc123...

# Optional (webhooks)
STRIPE_WEBHOOK_SECRET=whsec_abc123...

# Optional (resilience tuning)
STRIPE_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
STRIPE_CIRCUIT_BREAKER_TIMEOUT=60000
STRIPE_RETRY_MAX_RETRIES=3
```

---

## Troubleshooting

### Circuit Breaker Open
```typescript
// Check circuit breaker state
const bubble = new StripeBubble({...});
const state = bubble.resilience.getCircuitBreakerState();
console.log('Circuit breaker state:', state);

// Reset if needed (use sparingly)
await bubble.resilience.resetCircuitBreaker();
```

### Dead Letter Queue
```typescript
// Check for failed operations
const entries = bubble.resilience.getDeadLetterEntries();
console.log('Failed operations:', entries);

// Clear DLQ
bubble.resilience.clearDeadLetterQueue();
```

### Rate Limiting
```typescript
// Circuit breaker handles rate limiting automatically
// Monitor circuit breaker state in production
```

---

## Additional Resources

- **Full Implementation:** `stripe-bubble.ts` (1,293 lines)
- **Implementation Summary:** `STRIPE_BUBBLE_IMPLEMENTATION_SUMMARY.md`
- **Stripe API Docs:** https://stripe.com/docs/api
- **Stripe Testing:** https://stripe.com/docs/testing

---

**Quick Reference Created:** 2026-01-18
**Version:** 1.0.0
**Status:** Production Ready

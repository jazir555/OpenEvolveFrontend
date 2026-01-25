# Apify Service Bubble - Quick Reference

## File Location
```
BubbleLab/packages/bubble-core/src/bubbles/service-bubble/apify-bubble.ts
```

## Quick Stats
- **Total Lines:** 1,513
- **Operations:** 13 (12 required + 1 additional)
- **Test File:** apify-bubble.test.ts
- **Status:** ✅ Production Ready

## All Operations

### 1. runActor
Execute any Apify actor with custom input
```typescript
{
  operation: 'runActor',
  actorId: 'apify/web-scraper',
  input: { urls: ['https://example.com'] },
  memory: 1024,           // 128-8192 MB
  timeout: 300,           // 30-300 seconds
  waitForFinish: true,
  credentials: { APIFY_CRED: 'token' }
}
```

### 2. getActor
Retrieve actor details, versions, and statistics
```typescript
{
  operation: 'getActor',
  actorId: 'apify/web-scraper',
  credentials: { APIFY_CRED: 'token' }
}
```

### 3. listActors
Browse and discover available actors
```typescript
{
  operation: 'listActors',
  limit: 100,             // max 1000
  offset: 0,
  search: 'scraper',      // optional
  sortBy: 'createdAt',    // or 'modifiedAt', 'usageStats'
  credentials: { APIFY_CRED: 'token' }
}
```

### 4. buildActor
Build actor from source code
```typescript
{
  operation: 'buildActor',
  actorId: 'apify/web-scraper',
  buildTag: 'v1.0',       // optional
  version: '1.0.0',       // optional
  waitForFinish: true,
  credentials: { APIFY_CRED: 'token' }
}
```

### 5. getRun
Check status and details of an actor run
```typescript
{
  operation: 'getRun',
  runId: 'run-abc123xyz',
  credentials: { APIFY_CRED: 'token' }
}
```

### 6. waitForRun
Wait for run completion with polling
```typescript
{
  operation: 'waitForRun',
  runId: 'run-abc123xyz',
  waitFor: 300,           // max 3600 seconds
  waitInterval: 5,        // 1-60 seconds
  credentials: { APIFY_CRED: 'token' }
}
```

### 7. stopRun
Stop a running actor gracefully or immediately
```typescript
{
  operation: 'stopRun',
  runId: 'run-abc123xyz',
  gracefully: true,       // or false for immediate stop
  credentials: { APIFY_CRED: 'token' }
}
```

### 8. listRuns
List historical runs for an actor
```typescript
{
  operation: 'listRuns',
  actorId: 'apify/web-scraper',  // optional
  limit: 100,
  offset: 0,
  status: 'SUCCEEDED',   // optional filter
  credentials: { APIFY_CRED: 'token' }
}
```

### 9. getDataset
Get dataset metadata and information
```typescript
{
  operation: 'getDataset',
  datasetId: 'dataset-abc123',
  credentials: { APIFY_CRED: 'token' }
}
```

### 10. getDatasetItems
Fetch scraped data from datasets
```typescript
{
  operation: 'getDatasetItems',
  datasetId: 'dataset-abc123',
  limit: 1000,            // max 10000
  offset: 0,
  clean: true,
  format: 'json',         // or 'csv', 'xml', 'xlsx', 'html'
  credentials: { APIFY_CRED: 'token' }
}
```

### 11. downloadDataset
Download dataset in various formats
```typescript
{
  operation: 'downloadDataset',
  datasetId: 'dataset-abc123',
  format: 'json',         // or 'csv', 'xlsx', 'html'
  credentials: { APIFY_CRED: 'token' }
}
```

### 12. webScrape
Quick web scraping with selectors
```typescript
{
  operation: 'webScrape',
  url: 'https://example.com',
  selectors: ['.title', '.content'],  // optional
  proxyConfiguration: {
    useApifyProxy: true,
    proxyGroups: ['RESIDENTIAL'],     // optional
    countryCode: 'US'                 // optional
  },
  credentials: { APIFY_CRED: 'token' }
}
```

### 13. crawlWebsite
Crawl entire websites with proxy support
```typescript
{
  operation: 'crawlWebsite',
  startUrls: ['https://example.com/page1'],
  maxPages: 100,           // max 10000
  proxyConfiguration: {
    useApifyProxy: true
  },
  credentials: { APIFY_CRED: 'token' }
}
```

## Security Features

### URL Validation (SSRF Protection)
- ✅ Blocks: localhost, 127.0.0.1, private IPs
- ✅ Allows: HTTP, HTTPS
- ✅ Rejects: file://, ftp://, etc.

### ID Validation
- ✅ Actor ID: `username/actor-name` format
- ✅ Run ID: 10+ alphanumeric characters

### Memory Limits
- ✅ Minimum: 128 MB
- ✅ Maximum: 8,192 MB

### Error Sanitization
- ✅ Removes tokens from error messages
- ✅ Prevents credential leakage

## Rate Limiting

### Default Limits
- Requests: 25/minute
- Retries: 3 attempts
- Backoff: Exponential with jitter

### Timeouts
- API calls: 30 seconds
- Actor runs: 30-300 seconds
- Wait for run: Up to 3600 seconds

## Proxy Configuration

### Apify Proxy Options
```typescript
{
  useApifyProxy: true,
  proxyGroups: ['RESIDENTIAL', 'DATACENTER'],
  countryCode: 'US'  // 2-letter ISO code
}
```

### Proxy Groups
- `RESIDENTIAL` - Residential IPs
- `DATACENTER` - Datacenter IPs
- `GOOGLE_SERP` - For Google searches

## Common Use Cases

### Web Scraping
```typescript
const bubble = new ApifyBubble({
  operation: 'webScrape',
  url: 'https://example.com',
  selectors: ['.product-title', '.price'],
  credentials: { APIFY_CRED: process.env.APIFY_API_TOKEN }
});

const result = await bubble.action();
console.log(result.data);
```

### Website Crawling
```typescript
const bubble = new ApifyBubble({
  operation: 'crawlWebsite',
  startUrls: ['https://example.com'],
  maxPages: 100,
  proxyConfiguration: { useApifyProxy: true },
  credentials: { APIFY_CRED: process.env.APIFY_API_TOKEN }
});

const result = await bubble.action();
console.log(result.pagesCrawled);
```

### Custom Actor
```typescript
const bubble = new ApifyBubble({
  operation: 'runActor',
  actorId: 'apify/instagram-scraper',
  input: {
    directUrls: ['https://www.instagram.com/natgeo/'],
    resultsType: 'posts',
    resultsLimit: 10
  },
  memory: 2048,
  waitForFinish: true,
  credentials: { APIFY_CRED: process.env.APIFY_API_TOKEN }
});

const result = await bubble.action();
```

### Download Results
```typescript
const bubble = new ApifyBubble({
  operation: 'downloadDataset',
  datasetId: 'dataset-abc123',
  format: 'csv',
  credentials: { APIFY_CRED: process.env.APIFY_API_TOKEN }
});

const result = await bubble.action();
// result.content contains the CSV data
```

## Error Handling

### Common Errors
- `Invalid actor ID format` - Actor ID must be `username/name`
- `Invalid or unsafe URL` - URL blocked by SSRF protection
- `Run did not finish within timeout` - Increase waitFor value
- `Failed to start actor: 404` - Actor not found
- `Failed to start actor: 402` - Insufficient credits

### Retry Logic
- Transient errors: Automatic retry with backoff
- 429 (Rate Limit): Waits for Retry-After header
- 5xx errors: Retries up to 3 times

## Testing

### Run Tests
```bash
cd BubbleLab/packages/bubble-core
npx vitest run src/bubbles/service-bubble/apify-bubble.test.ts
```

### Test Coverage
- ✅ Schema validation for all operations
- ✅ Security validation
- ✅ Memory range validation
- ✅ Proxy configuration
- ✅ Timeout handling
- ✅ Error scenarios

## Environment Variables

```bash
APIFY_API_TOKEN=your-token-here
```

Get your token from: https://console.apify.com/

## Performance Tips

1. **Use waitForFinish: false** for long-running actors
   ```typescript
   { operation: 'runActor', waitForFinish: false }
   ```

2. **Adjust memory** for large datasets
   ```typescript
   { operation: 'runActor', memory: 4096 }
   ```

3. **Use proxy** for anti-scraping protection
   ```typescript
   { proxyConfiguration: { useApifyProxy: true } }
   ```

4. **Limit results** to reduce costs
   ```typescript
   { operation: 'getDatasetItems', limit: 100 }
   ```

## Support

- **Documentation:** https://docs.apify.com/
- **Console:** https://console.apify.com/
- **Actor Store:** https://apify.com/store
- **Issues:** Check implementation status in bubble-core repo

## Status

✅ **Production Ready** - All 12 operations complete with full security and error handling.

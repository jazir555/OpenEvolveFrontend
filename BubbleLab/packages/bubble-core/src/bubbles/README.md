# BubbleLab Bubbles Documentation

**Last Updated:** 2025-01-18
**Version:** 1.0.0

## Overview

This directory contains all BubbleLab bubbles - reusable components that encapsulate specific functionality. Bubbles are organized into three main categories:

- **Service Bubbles** - Integrations with external APIs and services
- **Tool Bubbles** - Utility functions and data processing tools
- **Workflow Bubbles** - Pre-built workflows that orchestrate multiple bubbles

## Directory Structure

```
bubbles/
├── service-bubble/          # External service integrations
│   ├── ai-agent.ts         # AI agent with tool support
│   ├── http.ts             # HTTP client
│   ├── slack.ts            # Slack integration
│   ├── github.ts           # GitHub API
│   ├── google-sheets/      # Google Sheets integration
│   ├── notion/             # Notion integration
│   └── apify/              # Apify actor integrations
├── tool-bubble/             # Utility and processing tools
│   ├── code-edit-tool.ts   # Code editing tool
│   ├── web-scrape-tool.ts  # Web scraping
│   ├── data-transformer-tool.ts
│   └── ...
└── workflow-bubble/         # Pre-built workflows
    ├── etl-pipeline.workflow.ts
    ├── slack-notifier.workflow.ts
    └── ...
```

## Service Bubbles

### AI & Machine Learning

#### AI Agent (`ai-agent.ts`)
**Purpose:** Multi-model AI agent with tool support and streaming

**Features:**
- Support for OpenAI, Anthropic, Google Gemini, OpenRouter, DeepSeek
- Tool calling with pre-registered or custom tools
- Streaming responses with thinking tokens
- Conversation history with KV cache optimization
- JSON mode for structured output
- Backup model support for reliability

**Usage:**
```typescript
const agent = new AIAgentBubble({
  message: 'What is the weather in Tokyo?',
  model: {
    model: 'openai/gpt-4o',
    temperature: 0.7,
    maxTokens: 4000
  },
  tools: [
    { name: 'web-search-tool' }
  ],
  credentials: {
    [CredentialType.OPENAI_CRED]: process.env.OPENAI_API_KEY
  }
});

const result = await agent.action();
console.log(result.response);
```

**Configuration:**
- Requires `OPENAI_CRED`, `ANTHROPIC_CRED`, `GOOGLE_GEMINI_CRED`, `OPENROUTER_CRED`, or `DEEPSEEK_CRED`
- Supports custom system prompts
- Configurable max iterations (default: 40)
- Optional streaming callback

**See Also:** [AI Agent Documentation](./service-bubble/ai-agent.ts)

### Communication

#### Slack (`slack.ts`)
**Purpose:** Comprehensive Slack API integration

**Features:**
- Send messages with attachments and blocks
- List and manage channels
- Get user and channel information
- Retrieve conversation history
- Add/remove reactions
- Upload files
- Update and delete messages

**Operations:**
- `send_message` - Send message to channel or DM
- `list_channels` - List workspace channels
- `get_channel_info` - Get channel details
- `list_users` - List workspace users
- `get_user_info` - Get user details
- `get_conversation_history` - Retrieve messages
- `get_thread_replies` - Get thread replies
- `update_message` - Edit existing message
- `delete_message` - Delete a message
- `add_reaction` - Add emoji reaction
- `remove_reaction` - Remove emoji reaction
- `join_channel` - Join a channel
- `upload_file` - Upload file to channel

**Usage:**
```typescript
const slack = new SlackBubble({
  operation: 'send_message',
  channel: 'general',
  text: 'Hello from BubbleLab!',
  blocks: [
    {
      type: 'section',
      text: {
        type: 'plain_text',
        text: 'Hello from BubbleLab!'
      }
    }
  ],
  credentials: {
    [CredentialType.SLACK_CRED]: process.env.SLACK_TOKEN
  }
});

const result = await slack.action();
```

**See Also:** [Slack API Documentation](https://api.slack.com/)

#### HTTP Client (`http.ts`)
**Purpose:** Generic HTTP client for external API calls

**Features:**
- All HTTP methods (GET, POST, PUT, PATCH, DELETE, etc.)
- Custom headers and authentication
- Request/response logging
- Timeout and retry support
- JSON auto-parsing
- SSRF protection

**Usage:**
```typescript
const http = new HttpBubble({
  url: 'https://api.example.com/data',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: {
    key: 'value'
  },
  timeout: 30000,
  credentials: {
    [CredentialType.CUSTOM_AUTH_KEY]: process.env.API_KEY
  }
});

const result = await http.action();
console.log(result.json);
```

**Security:**
- Blocks internal IP ranges
- Prevents SSRF via redirects
- Validates URL protocols
- Limits request body size

### Data Storage

#### PostgreSQL (`postgresql.ts`)
**Purpose:** PostgreSQL database operations

**Features:**
- Execute SQL queries
- Parameterized queries (SQL injection protection)
- Transaction support
- Connection pooling
- Query result streaming

**Usage:**
```typescript
const postgres = new PostgreSQLBubble({
  operation: 'query',
  query: 'SELECT * FROM users WHERE id = $1',
  params: [123],
  credentials: {
    [CredentialType.POSTGRES_CRED]: process.env.DATABASE_URL
  }
});

const result = await postgres.action();
```

#### Redis (`redis.ts`)
**Purpose:** Redis cache operations

**Features:**
- Get/set/delete operations
- TTL management
- Batch operations
- Pub/sub support

#### Elasticsearch (`elasticsearch.ts`)
**Purpose:** Elasticsearch search and indexing

**Features:**
- Full-text search
- Document indexing
- Query DSL support
- Aggregations

### File Management

#### Google Drive (`google-drive.ts`)
**Purpose:** Google Drive file operations

**Features:**
- List files and folders
- Upload/download files
- Create folders
- Share files
- Search files

#### Google Sheets (`google-sheets/`)
**Purpose:** Google Sheets operations

**Features:**
- Read/write cell values
- Batch operations
- Format cells
- Create spreadsheets
- Add sheets

**Files:**
- `google-sheets.ts` - Main API wrapper
- `google-sheets.schema.ts` - Zod schemas
- `google-sheets.utils.ts` - Utility functions
- `google-sheets.integration.flow.ts` - Integration workflow

### Email

#### Gmail (`gmail.ts`)
**Purpose:** Gmail API integration

**Features:**
- Send emails
- List messages
- Get message details
- Manage labels
- Search emails

#### SendGrid (`sendgrid.ts`)
**Purpose:** SendGrid email service

**Features:**
- Send emails
- Templates
- Attachments
- Batch sending

#### Resend (`resend.ts`)
**Purpose:** Resend email service

**Features:**
- Send emails
- HTML/templates
- Attachments

### Web & Scraping

#### Firecrawl (`firecrawl.ts`)
**Purpose:** Advanced web scraping with Firecrawl API

**Features:**
- Scrape single pages
- Crawl websites
- Extract structured data
- Sitemap support

#### Apify (`apify/`)
**Purpose:** Apify actor integrations

**Actors:**
- `google-maps-scraper` - Scrape Google Maps
- `instagram-scraper` - Scrape Instagram posts
- `instagram-hashtag-scraper` - Scrape hashtag feeds
- `linkedin-jobs-scraper` - Scrape LinkedIn jobs
- `linkedin-posts-search` - Search LinkedIn posts
- `linkedin-profile-posts` - Scrape LinkedIn profile posts
- `tiktok-scraper` - Scrape TikTok
- `twitter-scraper` - Scrape Twitter/X
- `youtube-scraper` - Scrape YouTube
- `youtube-transcript-scraper` - Extract YouTube transcripts

**Usage:**
```typescript
const scraper = new ApifyActorBubble({
  actorId: 'apify/google-maps-scraper',
  input: {
    search: 'restaurants near me',
    maxCrawledPlaces: 10
  },
  credentials: {
    [CredentialType.APIFY_CRED]: process.env.APIFY_TOKEN
  }
});

const result = await scraper.action();
```

### Productivity

#### Notion (`notion/`)
**Purpose:** Notion API integration

**Features:**
- Create pages
- Update pages
- Query databases
- Add blocks
- Property schemas

**Files:**
- `notion.ts` - Main API wrapper
- `property-schemas.ts` - Notion property schemas
- `index.ts` - Exports

#### GitHub (`github.ts`)
**Purpose:** GitHub API integration

**Features:**
- Repository operations
- Issues and PRs
- File operations
- Webhooks

### Payment

#### Stripe (`stripe.ts`)
**Purpose:** Stripe payment processing

**Features:**
- Create charges
- Manage customers
- Subscriptions
- Invoices

### Messaging

#### Twilio (`twilio.ts`)
**Purpose:** Twilio SMS and messaging

**Features:**
- Send SMS
- MMS support
- Media URLs

#### Telegram (`telegram.ts`)
**Purpose:** Telegram Bot API

**Features:**
- Send messages
- Get updates
- Inline keyboards

### Other Services

#### ElevenLabs (`eleven-labs.ts`)
**Purpose:** Text-to-speech synthesis

**Features:**
- Generate speech
- Voice selection
- SSML support

#### FollowUpBoss (`followupboss.ts`)
**Purpose:** Real estate CRM integration

#### Airtable (`airtable.ts`)
**Purpose:** Airtable database operations

## Tool Bubbles

### Code & Development

#### Code Edit Tool (`code-edit-tool.ts`)
**Purpose:** Apply code edits using Morph Fast Apply

**Features:**
- Intelligent code merging
- Lazy edit support with `// ... existing code ...` markers
- Validates code for security issues
- Returns diff of changes

**Usage:**
```typescript
const editor = new EditBubbleFlowTool({
  initialCode: 'function foo() { return 1; }',
  instructions: 'I am adding a bar function',
  codeEdit: `function foo() { return 1; }

// ... existing code ...

function bar() { return 2; }`,
  credentials: {
    [CredentialType.OPENROUTER_CRED]: process.env.OPENROUTER_API_KEY
  }
});

const result = await editor.performAction();
console.log(result.mergedCode);
```

**Security:**
- Blocks malicious patterns (eval, exec, etc.)
- Validates code size limits
- Sanitizes output

#### Code Formatter Tool (`code-formatter-tool.ts`)
**Purpose:** Format code with Prettier

#### BubbleFlow Validation Tool (`bubbleflow-validation-tool.ts`)
**Purpose:** Validate BubbleFlow definitions

### Data Processing

#### Data Transformer Tool (`data-transformer-tool.ts`)
**Purpose:** Transform and map data structures

**Features:**
- Field mapping
- Type conversion
- Conditional logic
- Array operations

#### CSV Processor Tool (`csv-processor-tool.ts`)
**Purpose:** Parse and generate CSV files

**Features:**
- Parse CSV to JSON
- Generate CSV from JSON
- Custom delimiters
- Header handling

#### JSON Validator Tool (`json-validator-tool.ts`)
**Purpose:** Validate JSON against schemas

#### XML Parser Tool (`xml-parser-tool.ts`)
**Purpose:** Parse XML to JSON

### Text Analysis

#### Text Analyzer Tool (`text-analyzer-tool.ts`)
**Purpose:** Analyze text content

**Features:**
- Sentiment analysis
- Keyword extraction
- Entity recognition
- Language detection

#### Log Parser Tool (`log-parser-tool.ts`)
**Purpose:** Parse log files

**Features:**
- Extract log levels
- Parse timestamps
- Identify patterns
- Filter by criteria

### File Processing

#### File Processor Tool (`file-processor-tool.ts`)
**Purpose:** Process file uploads

**Features:**
- Validate file types
- Size limits
- Content extraction

#### Image Processor Tool (`image-processor-tool.ts`)
**Purpose:** Process images

**Features:**
- Resize images
- Format conversion
- Compression
- Watermarking

#### PDF Generator Tool (`pdf-generator-tool.ts`)
**Purpose:** Generate PDF documents

**Features:**
- From HTML/templates
- Custom styling
- Headers/footers

### Web Tools

#### Web Search Tool (`web-search-tool.ts`)
**Purpose:** Search the web

**Features:**
- Search engines integration
- Result filtering
- Pagination

#### Web Scrape Tool (`web-scrape-tool.ts`)
**Purpose:** Scrape web pages

**Features:**
- HTML parsing
- CSS selectors
- Data extraction

#### Web Crawl Tool (`web-crawl-tool.ts`)
**Purpose:** Crawl websites

**Features:**
- Multi-page crawling
- Depth control
- URL filtering

#### Web Extract Tool (`web-extract-tool.ts`)
**Purpose:** Extract structured data from web pages

**Features:**
- LLM-based extraction
- Schema validation
- Batch processing

#### URL Validator Tool (`url-validator-tool.ts`)
**Purpose:** Validate and sanitize URLs

### Social Media

#### Instagram Tool (`instagram-tool.ts`)
**Purpose:** Instagram data extraction

**Features:**
- Post data
- User profiles
- Media downloads

#### LinkedIn Tool (`linkedin-tool.ts`)
**Purpose:** LinkedIn data extraction

**Features:**
- Profile data
- Job postings
- Company info

#### Twitter Tool (`twitter-tool.ts`)
**Purpose:** Twitter/X data extraction

**Features:**
- Tweet data
- User profiles
- Trends

#### TikTok Tool (`tiktok-tool.ts`)
**Purpose:** TikTok data extraction

**Features:**
- Video data
- User profiles
- Hashtag info

#### YouTube Tool (`youtube-tool.ts`)
**Purpose:** YouTube data extraction

**Features:**
- Video data
- Transcripts
- Comments

#### Reddit Scrape Tool (`reddit-scrape-tool.ts`)
**Purpose:** Reddit data extraction

**Features:**
- Post data
- Comments
- Subreddit info

### Validation

#### Email Validator Tool (`email-validator-tool.ts`)
**Purpose:** Validate email addresses

**Features:**
- Format validation
- Domain check
- Disposable email detection

### Utilities

#### Get Bubble Details Tool (`get-bubble-details-tool.ts`)
**Purpose:** Get bubble metadata

#### List Bubbles Tool (`list-bubbles-tool.ts`)
**Purpose:** List available bubbles

#### Metrics Collector Tool (`metrics-collector-tool.ts`)
**Purpose:** Collect execution metrics

#### Research Agent Tool (`research-agent-tool.ts`)
**Purpose:** AI-powered research assistant

#### SQL Query Tool (`sql-query-tool.ts`)
**Purpose:** Execute SQL queries

#### Vector Search Tool (`vector-search-tool.ts`)
**Purpose:** Semantic vector search

## Workflow Bubbles

Workflows are pre-built orchestrations that combine multiple bubbles.

### Data Workflows

#### ETL Pipeline (`etl-pipeline.workflow.ts`)
**Purpose:** Extract, Transform, Load data pipeline

**Steps:**
1. Extract data from source
2. Transform data
3. Load to destination

**Usage:**
```typescript
const workflow = new ETLWorkflow({
  source: {
    type: 'api',
    url: 'https://api.example.com/data'
  },
  transform: {
    mappings: {
      'old_field': 'new_field'
    }
  },
  destination: {
    type: 'database',
    table: 'processed_data'
  }
});

await workflow.execute();
```

#### Data Enrichment (`data-enrichment.workflow.ts`)
**Purpose:** Enrich data with additional information

**Steps:**
1. Load base data
2. Fetch enrichment data
3. Merge and return

#### API Aggregator (`api-aggregator.workflow.ts`)
**Purpose:** Aggregate data from multiple APIs

**Steps:**
1. Fetch from multiple sources
2. Normalize responses
3. Combine results

### Communication Workflows

#### Slack Notifier (`slack-notifier.workflow.ts`)
**Purpose:** Send notifications to Slack

**Steps:**
1. Format message
2. Add attachments/blocks
3. Send to Slack

#### Slack Data Assistant (`slack-data-assistant.workflow.ts`)
**Purpose:** AI-powered Slack assistant

**Steps:**
1. Receive Slack message
2. Process with AI
3. Execute tools
4. Reply with results

#### Slack Formatter Agent (`slack-formatter-agent.ts`)
**Purpose:** Format data for Slack display

### Document Workflows

#### Generate Document (`generate-document.workflow.ts`)
**Purpose:** Generate documents from templates

#### Parse Document (`parse-document.workflow.ts`)
**Purpose:** Parse document content

#### PDF OCR (`pdf-ocr.workflow.ts`)
**Purpose:** Extract text from PDFs using OCR

#### PDF Form Operations (`pdf-form-operations.workflow.ts`)
**Purpose:** Fill and process PDF forms

### Monitoring Workflows

#### Monitoring & Alert (`monitoring-alert.workflow.ts`)
**Purpose:** Monitor and send alerts

**Steps:**
1. Check metrics
2. Evaluate conditions
3. Send alerts if needed

### Business Workflows

#### Multi-Step Approval (`multi-step-approval.workflow.ts`)
**Purpose:** Multi-level approval workflow

#### Backup & Restore (`backup-restore.workflow.ts`)
**Purpose:** Backup and restore data

#### Database Analyzer (`database-analyzer.workflow.ts`)
**Purpose:** Analyze database structure

#### Scheduled Task (`scheduled-task.workflow.ts`)
**Purpose:** Execute scheduled tasks

#### Event Handler (`event-handler.workflow.ts`)
**Purpose:** Handle webhook events

#### Webhook Repeater (`webhook-repeater.workflow.ts`)
**Purpose:** Repeat webhooks to multiple endpoints

## Credential Types

Bubbles use the following credential types (defined in `@bubblelab/shared-schemas`):

- `OPENAI_CRED` - OpenAI API key
- `ANTHROPIC_CRED` - Anthropic API key
- `GOOGLE_GEMINI_CRED` - Google Gemini API key
- `OPENROUTER_CRED` - OpenRouter API key
- `DEEPSEEK_CRED` - DeepSeek API key
- `SLACK_CRED` - Slack Bot Token
- `GITHUB_CRED` - GitHub Personal Access Token
- `GMAIL_CRED` - Gmail OAuth token
- `GOOGLE_SHEETS_CRED` - Google Sheets OAuth token
- `GOOGLE_DRIVE_CRED` - Google Drive OAuth token
- `POSTGRES_CRED` - PostgreSQL connection string
- `REDIS_CRED` - Redis connection string
- `ELASTICSEARCH_CRED` - Elasticsearch URL
- `TWILIO_CRED` - Twilio Account SID + Token
- `SENDGRID_CRED` - SendGrid API key
- `RESEND_CRED` - Resend API key
- `STRIPE_CRED` - Stripe API key
- `TELEGRAM_CRED` - Telegram Bot Token
- `NOTION_CRED` - Notion Integration Token
- `AIRTABLE_CRED` - Airtable Personal Access Token
- `APIFY_CRED` - Apify API token
- `FIRECRAWL_CRED` - Firecrawl API key
- `ELEVENLABS_CRED` - ElevenLabs API key
- `FOLLOWUPBOSS_CRED` - FollowUpBoss API key
- `CUSTOM_AUTH_KEY` - Custom API key

## Common Patterns

### Error Handling

```typescript
const bubble = new SomeBubble({ param: 'value' });

try {
  const result = await bubble.action();
  if (result.success) {
    console.log(result.data);
  } else {
    console.error(result.error);
  }
} catch (error) {
  console.error('Unexpected error:', error);
}
```

### Streaming

```typescript
const agent = new AIAgentBubble({
  message: 'Tell me a story',
  streaming: true,
  streamingCallback: async (event) => {
    if (event.type === 'llm_complete') {
      console.log(event.data.content);
    }
  }
});

await agent.action();
```

### Retry Logic

Most bubbles implement automatic retry logic. Configure via:

```typescript
const bubble = new SomeBubble({
  param: 'value',
  maxRetries: 5,
  retryDelay: 2000
});
```

### Tool Calling

```typescript
const agent = new AIAgentBubble({
  message: 'Search the web for latest news',
  tools: [
    { name: 'web-search-tool' },
    { name: 'web-scrape-tool' }
  ],
  credentials: {
    [CredentialType.OPENAI_CRED]: process.env.OPENAI_API_KEY
  }
});

const result = await agent.action();
console.log(result.toolCalls);
```

## Contributing

When adding new bubbles:

1. **Follow naming conventions:**
   - Service bubbles: `{service}-bubble.ts`
   - Tool bubbles: `{tool}-tool.ts`
   - Workflows: `{workflow}.workflow.ts`

2. **Include comprehensive JSDoc:**
   - File-level documentation
   - Method documentation
   - Type documentation
   - Usage examples

3. **Implement proper error handling:**
   - Validate inputs
   - Handle API errors gracefully
   - Return structured error messages

4. **Add credentials to shared schemas:**
   - Define credential type
   - Add to `BUBBLE_CREDENTIAL_OPTIONS`
   - Document credential format

5. **Write tests:**
   - Unit tests for logic
   - Integration tests for API calls
   - Mock external dependencies

## Troubleshooting

### Common Issues

**Issue: Authentication errors**
- Solution: Verify credentials are correct and properly injected
- Check credential type matches bubble requirements

**Issue: Timeouts**
- Solution: Increase `timeout` parameter
- Check network connectivity
- Verify API endpoint is reachable

**Issue: Rate limiting**
- Solution: Implement exponential backoff
- Reduce request frequency
- Check API quota

**Issue: Invalid responses**
- Solution: Validate schema matches API documentation
- Check API version compatibility
- Review response parsing logic

### Debug Mode

Enable debug logging:

```typescript
process.env.DEBUG = 'bubblelab:*';
const bubble = new SomeBubble({ param: 'value' });
```

### Testing Credentials

Test if credentials are valid:

```typescript
const bubble = new SomeBubble({ param: 'value' });
const isValid = await bubble.testCredential();
console.log('Credentials valid:', isValid);
```

## Additional Resources

- [BubbleLab Main Documentation](../../README.md)
- [Shared Schemas](../../shared-schemas/)
- [Bubble Factory](../../bubble-factory.ts)
- [Examples](../../../examples/)

## License

MIT

---

**Last Updated:** 2025-01-18
**Maintainers:** BubbleLab Team

# Tool Bubbles - Quick Start Guide

## 🚀 Quick Implementation Reference

### Completed Tools (Copy & Use Now)

#### ✅ WebSearchTool
```typescript
import { WebSearchTool } from './tool-bubble/web-search-tool';

// Using DuckDuckGo (Free - No API Key)
const search = new WebSearchTool({
  apiProvider: 'duckduckgo',
  maxResults: 10
});

const results = await search.search({
  query: 'TypeScript best practices'
});

// Using SerpAPI (Requires API Key)
const searchAPI = new WebSearchTool({
  apiKey: process.env.SERPAPI_API_KEY,
  apiProvider: 'serpapi'
});

const apiResults = await searchAPI.searchNews({
  query: 'AI news',
  num: 20
});
```

#### ✅ SQLQueryTool
```typescript
import { SQLQueryTool } from './tool-bubble/sql-query-tool';

const sql = new SQLQueryTool({
  maxResults: 1000
});

// Validate query
const validation = await sql.validate({
  sql: 'SELECT * FROM users WHERE active = true LIMIT 10'
});

// Format query
const formatted = await sql.format({
  sql: 'select id,name from users where active=true'
});

// Execute query
const results = await sql.query({
  sql: 'SELECT * FROM users LIMIT 10'
});
```

---

### Implementation Templates (Copy These Patterns)

#### Template 1: API Integration Tool
```typescript
import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';
import axios from 'axios';

export class ExampleAPITool extends ToolBubble<Params, Result> {
  bubbleName = 'example-api';
  type = 'tool';
  alias = 'example-api';

  private apiKey: string;
  private baseUrl: string;

  params = {
    apiKey: z.string().optional(),
    timeout: z.number().int().positive().default(30000)
  };

  constructor(params: Params = {}) {
    super(params);
    this.apiKey = params.apiKey || process.env.EXAMPLE_API_KEY || '';
    this.baseUrl = 'https://api.example.com';
  }

  async execute(input: any): Promise<Result> {
    try {
      if (!input.endpoint) {
        throw new Error('Endpoint is required');
      }

      const data = await this.makeRequest(input.endpoint, {
        method: 'GET',
        params: input.params
      });

      return { success: true, data };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  private async makeRequest(
    endpoint: string,
    options?: AxiosRequestConfig
  ): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;

    const response = await axios(url, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        ...options?.headers
      },
      timeout: this.params.timeout.default()
    });

    return response.data;
  }
}

interface Params {
  apiKey?: string;
  timeout?: number;
}

interface Result {
  success: boolean;
  data?: any;
  error?: string;
  timestamp?: string;
}
```

#### Template 2: File Processing Tool
```typescript
import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';
import * as fs from 'fs-extra';

export class ExampleFileTool extends ToolBubble<Params, Result> {
  bubbleName = 'example-file';
  type = 'tool';
  alias = 'example-file';

  params = {
    maxFileSize: z.number().int().positive().default(10485760) // 10MB
  };

  async execute(input: any): Promise<Result> {
    try {
      if (!input.path) {
        throw new Error('Path is required');
      }

      // Validate file size
      const stats = await fs.stat(input.path);
      if (stats.size > this.params.maxFileSize.default()) {
        throw new Error('File too large');
      }

      // Process file
      const content = await fs.readFile(input.path, 'utf-8');

      return {
        success: true,
        data: content,
        size: stats.size
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        path: input.path
      };
    }
  }
}

interface Params {
  maxFileSize?: number;
}

interface Result {
  success: boolean;
  data?: any;
  size?: number;
  error?: string;
  path?: string;
}
```

#### Template 3: Data Transformation Tool
```typescript
import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

export class ExampleTransformTool extends ToolBubble<Params, Result> {
  bubbleName = 'example-transform';
  type = 'tool';
  alias = 'example-transform';

  async execute(input: any): Promise<Result> {
    try {
      if (!input.data) {
        throw new Error('Data is required');
      }

      const transformed = await this.transform(input);

      return {
        success: true,
        transformed,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async transform(input: {
    data: any;
    operations: Array<{ type: string; params?: any }>;
  }): Promise<any> {
    let result = input.data;

    for (const op of input.operations) {
      switch (op.type) {
        case 'filter':
          result = result.filter(op.params);
          break;
        case 'map':
          result = result.map(op.params);
          break;
        case 'sort':
          result = result.sort(op.params);
          break;
        case 'reduce':
          result = result.reduce(op.params);
          break;
        default:
          throw new Error(`Unknown operation: ${op.type}`);
      }
    }

    return result;
  }
}

interface Params {}

interface Result {
  success: boolean;
  transformed?: any;
  error?: string;
  timestamp?: string;
}
```

---

## 📋 Tool Implementation Checklist

For each tool, ensure:

### Basic Requirements
- [ ] Extends `ToolBubble` base class
- [ ] Has `bubbleName`, `type`, and `alias` properties
- [ ] Implements `params` with Zod validation
- [ ] Implements `execute()` method
- [ ] Returns proper result interface

### Error Handling
- [ ] Try-catch blocks in all async methods
- [ ] Meaningful error messages
- [ ] Timestamp on errors
- [ ] Input validation
- [ ] Null/undefined checks

### Security
- [ ] Input sanitization
- [ ] SQL injection prevention (if applicable)
- [ ] XSS prevention (if applicable)
- [ ] File size limits
- [ ] Timeout protection
- [ ] Rate limiting (if applicable)

### Testing
- [ ] Unit tests for all methods
- [ ] Error case tests
- [ ] Edge case tests
- [ ] Integration tests
- [ ] Performance tests

### Documentation
- [ ] JSDoc comments
- [ ] Usage examples
- [ ] Parameter descriptions
- [ ] Return value descriptions
- [ ] Error scenarios documented

---

## 🎯 Priority Implementation Order

### Week 1: Foundation Tools
1. ✅ WebSearchTool (DONE)
2. ✅ SQLQueryTool (DONE)
3. WebScrapeTool
4. FileProcessorTool
5. MetricsCollectorTool

### Week 2: Data Processing
6. VectorSearchTool
7. TextAnalyzerTool
8. ImageProcessorTool
9. PDFGeneratorTool

### Week 3: Validation
10. EmailValidatorTool
11. URLValidatorTool
12. CodeFormatterTool

### Week 4: Advanced
13. ResearchAgentTool
14. GetBubbleDetailsTool
15. ListBubblesTool
16. ChartJSTool
17. GoogleMapsTool

### Week 5: Integration
18. Social Media Tools (5 platforms)

---

## 🔧 Common Patterns

### Retry Logic
```typescript
private async retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt === maxRetries - 1) throw error;
      await new Promise(resolve =>
        setTimeout(resolve, Math.pow(2, attempt) * 1000)
      );
    }
  }
  throw new Error('Max retries exceeded');
}
```

### Validation Helper
```typescript
private validateRequired(params: any, fields: string[]): void {
  const missing = fields.filter(field => !params[field]);
  if (missing.length > 0) {
    throw new Error(`Missing required fields: ${missing.join(', ')}`);
  }
}
```

### Response Formatter
```typescript
private formatResponse(success: boolean, data: any, error?: string) {
  return {
    success,
    ...(success ? { data } : { error }),
    timestamp: new Date().toISOString()
  };
}
```

---

## 📦 Quick Install Commands

### All Dependencies
```bash
cd BubbleLab/packages/bubble-core

# Core
npm install axios cheerio

# Database
npm install pg mysql2 better-sqlite3 tedious

# Files
npm install fs-extra csv-parser csv-writer xlsx

# Images
npm install sharp

# PDF
npm install pdfkit jspdf pdf-lib

# NLP
npm install compromise natural sentiment franc

# Validation
npm install email-validator disposable-email-domains is-reachable

# Code
npm install prettier eslint

# Charts
npm install chart.js chartjs-node-canvas

# Maps
npm install @googlemaps/google-maps-services-js

# Vector DB
npm install @pinecone-database/pinecone weaviate-ts-client openai

# Metrics
npm install prom-client @influxdata/influxdb-client redis
```

### By Tool Category
```bash
# Web scraping
npm install axios cheerio puppeteer

# File operations
npm install fs-extra csv-parser csv-writer

# Image processing
npm install sharp

# PDF generation
npm install pdfkit jspdf

# NLP/Text analysis
npm install compromise natural sentiment

# Email/URL validation
npm install email-validator is-reachable
```

---

## 🧪 Testing Template

```typescript
import { YourTool } from './your-tool';

describe('YourTool', () => {
  let tool: YourTool;

  beforeEach(() => {
    tool = new YourTool();
  });

  describe('execute()', () => {
    it('should succeed with valid input', async () => {
      const result = await tool.execute({
        /* valid input */
      });

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });

    it('should fail with invalid input', async () => {
      const result = await tool.execute({
        /* invalid input */
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should handle missing required fields', async () => {
      const result = await tool.execute({});

      expect(result.success).toBe(false);
      expect(result.error).toContain('required');
    });
  });

  describe('error handling', () => {
    it('should return error message on failure', async () => {
      const result = await tool.execute({
        /* input that causes error */
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeInstanceOf(String);
      expect(result.error.length).toBeGreaterThan(0);
    });

    it('should include timestamp on error', async () => {
      const result = await tool.execute({
        /* input that causes error */
      });

      if (!result.success) {
        expect(result.timestamp).toBeDefined();
      }
    });
  });
});
```

---

## 🎓 Resources

### Documentation
- `/docs/TOOL_BUBBLE_IMPLEMENTATION_STATUS.md` - Detailed status
- `/docs/TOOL_BUBBLE_DEPENDENCIES.md` - Dependencies
- `/docs/TOOL_BUBBLE_IMPLEMENTATION_GUIDE.md` - Code examples
- `/docs/TOOL_BUBBLE_IMPLEMENTATION_SUMMARY.md` - Executive summary

### Tool Locations
- Source: `/BubbleLab/packages/bubble-core/src/bubbles/tool-bubble/`
- Tests: `/BubbleLab/packages/bubble-core/src/bubbles/tool-bubble/__tests__/`

### Environment Variables
See `/docs/TOOL_BUBBLE_DEPENDENCIES.md` for complete `.env` setup.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
npm install axios cheerio fs-extra csv-parser csv-writer sharp
```

### 2. Implement Your Tool
```typescript
// Copy template, customize for your needs
export class MyTool extends ToolBubble {
  bubbleName = 'my-tool';
  type = 'tool';
  alias = 'my-tool';

  async execute(input: any) {
    // Your implementation
  }
}
```

### 3. Test It
```bash
npm test -- my-tool.test.ts
```

### 4. Use It
```typescript
const tool = new MyTool();
const result = await tool.execute({ /* input */ });
```

---

## ✅ Success Criteria

Your tool is production-ready when:

- ✅ No placeholder code
- ✅ Real library/service integration
- ✅ Comprehensive error handling
- ✅ Input validation
- ✅ Security checks
- ✅ Unit tests (80%+ coverage)
- ✅ Documentation complete
- ✅ Examples provided

---

**Happy Coding! 🎉**

*Last Updated: 2025-01-17*
*BubbleLab Tool Bubbles v1.0.0*

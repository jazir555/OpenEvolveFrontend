# Tool Bubble Dependencies - Package Configuration

## Overview

This document lists all dependencies required to implement the 18 critical tool bubbles with real, production-ready integrations.

## Complete package.json Additions

Add these dependencies to your `bubble-core/package.json`:

```json
{
  "dependencies": {
    "=== Current Dependencies ===": "",
    "zod": "^3.22.4",

    "=== Web Scraping & HTTP ===": "",
    "axios": "^1.6.2",
    "cheerio": "^1.0.0-rc.12",
    "puppeteer": "^21.6.1",
    "node-fetch": "^3.3.2",

    "=== Database Clients ===": "",
    "pg": "^8.11.3",
    "mysql2": "^3.6.5",
    "better-sqlite3": "^9.2.2",
    "tedious": "^16.6.1",

    "=== Vector Database ===": "",
    "@pinecone-database/pinecone": "^1.1.3",
    "weaviate-ts-client": "^1.5.0",
    "openai": "^4.20.1",

    "=== Metrics & Monitoring ===": "",
    "prom-client": "^15.1.0",
    "@influxdata/influxdb-client": "^1.33.2",
    "redis": "^4.6.11",

    "=== File Processing ===": "",
    "fs-extra": "^11.2.0",
    "csv-parser": "^3.0.0",
    "fast-csv": "^4.3.6",
    "xlsx": "^0.18.5",

    "=== Image Processing ===": "",
    "sharp": "^0.33.1",

    "=== PDF Generation ===": "",
    "pdfkit": "^0.14.0",
    "jspdf": "^2.5.1",
    "pdf-lib": "^1.17.1",

    "=== Email & URL Validation ===": "",
    "email-validator": "^2.0.4",
    "disposable-email-domains": "^1.0.62",
    "is-reachable": "^5.2.1",

    "=== Code Formatting ===": "",
    "prettier": "^3.1.1",
    "eslint": "^8.56.0",

    "=== Text Analysis & NLP ===": "",
    "compromise": "^14.10.0",
    "natural": "^6.9.0",
    "sentiment": "^5.0.2",
    "franc": "^6.1.0",

    "=== Charting & Visualization ===": "",
    "chart.js": "^4.4.1",
    "chartjs-node-canvas": "^4.1.6",

    "=== Google Services ===": "",
    "@googlemaps/google-maps-services-js": "^3.10.1",

    "=== Social Media APIs ===": "",
    "instagram-url-direct": "^1.0.11",
    "linkedin-api": "^1.0.0",
    "twitter-api-v2": "^1.15.0",
    "tiktok-api": "^2.11.0",
    "snoowrap": "^1.23.0"
  },
  "devDependencies": {
    "@types/node": "^20.10.6",
    "@types/fs-extra": "^11.0.4",
    "@types/better-sqlite3": "^7.6.8",
    "@types/sharp": "^0.32.0",
    "@types/prettier": "^3.0.2",
    "@types/natural": "^6.9.0",
    "typescript": "^5.3.3"
  }
}
```

## Individual Tool Dependencies

### 1. WebSearchTool ✅ (No additional dependencies)
- Uses native `fetch` API
- Environment variables for API keys

### 2. SQLQueryTool ✅ (Add for full database support)
```json
{
  "dependencies": {
    "pg": "^8.11.3",           // PostgreSQL
    "mysql2": "^3.6.5",        // MySQL
    "better-sqlite3": "^9.2.2", // SQLite
    "tedious": "^16.6.1"       // SQL Server
  }
}
```

### 3. WebScrapeTool
```json
{
  "dependencies": {
    "axios": "^1.6.2",         // HTTP client
    "cheerio": "^1.0.0-rc.12", // HTML parsing
    "puppeteer": "^21.6.1"     // JavaScript rendering (optional)
  }
}
```

### 4. ResearchAgentTool
```json
{
  "dependencies": {
    "openai": "^4.20.1"        // LLM integration
  }
}
```

### 5. VectorSearchTool
```json
{
  "dependencies": {
    "@pinecone-database/pinecone": "^1.1.3", // Pinecone
    "weaviate-ts-client": "^1.5.0",          // Weaviate
    "openai": "^4.20.1"                       // Embeddings
  }
}
```

### 6. MetricsCollectorTool
```json
{
  "dependencies": {
    "prom-client": "^15.1.0",                 // Prometheus
    "@influxdata/influxdb-client": "^1.33.2", // InfluxDB
    "redis": "^4.6.11"                        // Redis
  }
}
```

### 7. FileProcessorTool
```json
{
  "dependencies": {
    "fs-extra": "^11.2.0",  // Enhanced file operations
    "csv-parser": "^3.0.0", // CSV parsing
    "fast-csv": "^4.3.6",   // CSV writing
    "xlsx": "^0.18.5"       // Excel support
  }
}
```

### 8. ImageProcessorTool
```json
{
  "dependencies": {
    "sharp": "^0.33.1"  // High-performance image processing
  }
}
```

### 9. PDFGeneratorTool
```json
{
  "dependencies": {
    "pdfkit": "^0.14.0",   // PDF generation
    "jspdf": "^2.5.1",     // Browser-based PDF
    "pdf-lib": "^1.17.1"   // PDF manipulation
  }
}
```

### 10. EmailValidatorTool
```json
{
  "dependencies": {
    "email-validator": "^2.0.4",              // Email validation
    "disposable-email-domains": "^1.0.62"     // Disposable email detection
  }
}
```

### 11. URLValidatorTool
```json
{
  "dependencies": {
    "is-reachable": "^5.2.1"  // URL reachability
  }
}
```

### 12. CodeFormatterTool
```json
{
  "dependencies": {
    "prettier": "^3.1.1"  // Code formatting
  }
}
```

### 13. TextAnalyzerTool
```json
{
  "dependencies": {
    "compromise": "^14.10.0",  // NLP
    "natural": "^6.9.0",       // General NLP
    "sentiment": "^5.0.2",     // Sentiment analysis
    "franc": "^6.1.0"          // Language detection
  }
}
```

### 14. ChartJSTool
```json
{
  "dependencies": {
    "chart.js": "^4.4.1",             // Charting library
    "chartjs-node-canvas": "^4.1.6"   // Server-side rendering
  }
}
```

### 15. GoogleMapsTool
```json
{
  "dependencies": {
    "@googlemaps/google-maps-services-js": "^3.10.1"
  }
}
```

### 16. Social Media Tools
```json
{
  "dependencies": {
    "instagram-url-direct": "^1.0.11",  // Instagram
    "linkedin-api": "^1.0.0",           // LinkedIn
    "twitter-api-v2": "^1.15.0",        // Twitter
    "tiktok-api": "^2.11.0",            // TikTok
    "snoowrap": "^1.23.0"               // Reddit
  }
}
```

## Installation Commands

### Install All Dependencies
```bash
cd /c/Users/mmeadow/Documents/OpenEvolve/Frontend/docs/BubbleLab/packages/bubble-core
npm install pg mysql2 better-sqlite3 tedious
npm install axios cheerio puppeteer
npm install @pinecone-database/pinecone weaviate-ts-client openai
npm install prom-client @influxdata/influxdb-client redis
npm install fs-extra csv-parser fast-csv xlsx
npm install sharp
npm install pdfkit jspdf pdf-lib
npm install email-validator disposable-email-domains is-reachable
npm install prettier
npm install compromise natural sentiment franc
npm install chart.js chartjs-node-canvas
npm install @googlemaps/google-maps-services-js
npm install instagram-url-direct linkedin-api twitter-api-v2 tiktok-api snoowrap
```

### Install by Tool Category
```bash
# Web Scraping
npm install axios cheerio puppeteer

# Databases
npm install pg mysql2 better-sqlite3 tedious

# Vector Search
npm install @pinecone-database/pinecone weaviate-ts-client openai

# Metrics
npm install prom-client @influxdata/influxdb-client redis

# File Operations
npm install fs-extra csv-parser fast-csv xlsx

# Images
npm install sharp

# PDF
npm install pdfkit jspdf pdf-lib

# Validation
npm install email-validator disposable-email-domains is-reachable

# NLP
npm install compromise natural sentiment franc

# Charts
npm install chart.js chartjs-node-canvas

# Maps
npm install @googlemaps/google-maps-services-js

# Social Media
npm install instagram-url-direct linkedin-api twitter-api-v2 tiktok-api snoowrap
```

## Peer Dependencies

Some packages may require peer dependencies:

```bash
# For TypeScript types
npm install --save-dev @types/node @types/fs-extra @types/better-sqlite3 @types/sharp @types/prettier @types/natural
```

## Environment Setup

Create a `.env.example` file:

```bash
# Search APIs
SERPAPI_API_KEY=
GOOGLE_API_KEY=
GOOGLE_SEARCH_ENGINE_ID=
BING_API_KEY=

# Databases
DATABASE_URL=
POSTGRES_URL=
MYSQL_URL=
SQLITE_PATH=

# Vector Database
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
WEAVIATE_URL=

# OpenAI (for embeddings and LLM)
OPENAI_API_KEY=

# Metrics
PROMETHEUS_PORT=
INFLUXDB_URL=
INFLUXDB_TOKEN=
REDIS_URL=

# Maps
GOOGLE_MAPS_API_KEY=

# Social Media
INSTAGRAM_ACCESS_TOKEN=
INSTAGRAM_APP_ID=
INSTAGRAM_APP_SECRET=
LINKEDIN_ACCESS_TOKEN=
TWITTER_BEARER_TOKEN=
TWITTER_API_KEY=
TWITTER_API_SECRET=
TIKTOK_API_KEY=
TIKTOK_API_SECRET=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
```

## Version Notes

- **Node.js**: Requires Node.js 18+ or 20+
- **TypeScript**: Requires TypeScript 5.0+
- **Platform**: Some packages (like `sharp` and `better-sqlite3`) require native compilation

## Platform-Specific Notes

### Windows
```bash
# For native modules
npm install --global windows-build-tools
```

### macOS
```bash
# For native modules
xcode-select --install
```

### Linux
```bash
# For native modules
sudo apt-get install build-essential
```

## Testing Dependencies

```bash
npm install --save-dev \
  jest@^29.7.0 \
  @types/jest@^29.5.11 \
  ts-jest@^29.1.1 \
  @testing-library/jest-dom@^6.1.5
```

## Production Considerations

1. **Bundle Size**: Some packages are large (e.g., `puppeteer` ~300MB). Consider:
   - Using `puppeteer-core` instead of `puppeteer`
   - Loading heavy dependencies only when needed
   - Using tree-shaking

2. **Security**: Always use `npm audit` to check for vulnerabilities:
   ```bash
   npm audit
   npm audit fix
   ```

3. **Licenses**: Review all package licenses for production use:
   ```bash
   npm install -g license-checker
   license-checker
   ```

## Alternative Lightweight Options

If bundle size is a concern:

| Heavy Package | Lightweight Alternative |
|--------------|------------------------|
| puppeteer | axios + cheerio (no JS rendering) |
| sharp | jimp (pure JavaScript, slower) |
| pdfkit | pdf.js (lighter) |
| natural | compromise (smaller, focused) |

---

## Summary

- **Total dependencies to add**: ~40 packages
- **Estimated install size**: ~500MB-1GB (with all optional dependencies)
- **Core required size**: ~100MB (without Puppeteer)
- **Production recommendation**: Start with core dependencies, add others as needed

---

Next Steps:
1. Review and install required dependencies
2. Implement tools in priority order
3. Add comprehensive testing
4. Document API usage examples

# RAGBits BubbleLabs Plugin - Quick Start Guide

## Installation

```bash
cd bubblelabs-ragbits-plugin
npm install
```

## Development

```bash
# Run development server
npm run dev

# Run tests
npm test

# Build plugin
npm run build

# Lint code
npm run lint
```

## Integration Steps

### 1. Install Plugin in BubbleLabs

```bash
cd BubbleLab
npm install ../bubblelabs-ragbits-plugin
```

### 2. Add Plugin Import to BubbleLabs

In `BubbleLab/apps/bubble-studio/src/lib/integrations.ts`:

```typescript
import { RAGBitsPlugin } from 'bubblelabs-ragbits-plugin';

// Add to SERVICE_LOGOS
export const SERVICE_LOGOS = {
  // ...existing logos
  RAGBits: '/integrations/ragbits.svg',
};

// Add to OPENEVOLVE_INTEGRATIONS
export const OPENEVOLVE_INTEGRATIONS = [
  // ...existing integrations
  { name: 'RAGBits', file: SERVICE_LOGOS['RAGBits'] },
];
```

### 3. Use Plugin in Components

```typescript
import { createPlugin } from 'bubblelabs-ragbits-plugin';

// Initialize
const ragbitsPlugin = createPlugin({
  serverUrl: 'http://localhost:3000/ragbits'
});

// Search
const results = await ragbitsPlugin.search({
  query: 'authentication system',
  topK: 10
});
```

### 4. Add RAGBits Icon

Add `ragbits.svg` to `BubbleLab/apps/bubble-studio/public/integrations/`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
</svg>
```

## Plugin Features

### Search Capabilities
- Semantic vector search
- Hybrid keyword + semantic search
- Reranking for improved relevance
- Advanced filtering (type, stage, team, tags)

### Document Management
- Automatic artifact indexing
- Batch document ingestion
- Rich metadata support
- Index statistics monitoring

### Performance
- Configurable caching
- Batch processing
- Operation history tracking
- Performance metrics

## Configuration Options

```typescript
{
  serverUrl: string;              // RAGBits server URL
  apiKey?: string;                // Optional API key
  timeout?: number;               // Request timeout (seconds)
  defaultTopK: number;            // Default result count
  defaultScoreThreshold: number;  // Minimum relevance score
  enableHybridSearch: boolean;    // Enable hybrid search
  enableReranking: boolean;       // Enable result reranking
  autoIndexArtifacts: boolean;    // Auto-index workflow artifacts
  indexingBatchSize: number;      // Batch size for indexing
  enableCaching: boolean;         // Enable result caching
  cacheTTLSeconds: number;        // Cache TTL
  maxSearchTime: number;          // Max search time (seconds)
}
```

## Testing

```bash
# Test search functionality
curl -X POST http://localhost:3000/ragbits/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "topK": 5}'

# Test document ingestion
curl -X POST http://localhost:3000/ragbits/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "Test document", "metadata": {"type": "test"}}'
```

## Troubleshooting

### Plugin not loading
- Check npm installation
- Verify plugin import path
- Check browser console for errors

### Connection refused
- Verify RAGBits server is running
- Check serverUrl configuration
- Test server health endpoint

### No search results
- Verify documents are indexed
- Check scoreThreshold setting
- Try lowering scoreThreshold
- Enable debug mode for details

## Architecture

```
BubbleLabs UI
  ↓
Plugin Components (React)
  ↓
Plugin Factory (createRAGBitsPlugin)
  ↓
RAGBits Service (Caching + Business Logic)
  ↓
RAGBits Client (HTTP Communication)
  ↓
RAGBits Server (Vector Search API)
```

## File Structure

```
bubblelabs-ragbits-plugin/
├── src/
│   ├── types/           # TypeScript type definitions
│   ├── lib/             # Low-level HTTP client
│   ├── services/        # High-level service wrapper
│   ├── components/      # React UI components
│   ├── hooks/           # React hooks
│   ├── utils/           # Plugin factory
│   └── index.ts         # Main export
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Next Steps

1. Build the plugin: `npm run build`
2. Test with BubbleLabs
3. Configure RAGBits server connection
4. Index existing documents
5. Test search functionality
6. Integrate with workflow automation

## Support

For issues or questions:
- Check the main README.md
- Review example usage
- Open GitHub issue
- Contact OpenEvolve support

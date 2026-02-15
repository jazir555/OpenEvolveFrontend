# BubbleLabs RAGBits Plugin

![RAGBits Logo](https://via.placeholder.com/150/4A90E2/FFFFFF?text=RAGBits)

**Standalone plugin for BubbleLabs that integrates RAGBits' semantic document search and knowledge retrieval capabilities.**

## 📋 Overview

The BubbleLabs RAGBits Plugin provides a **zero-modification** integration that adds powerful semantic document search and knowledge retrieval to BubbleLabs. The plugin is **fully configurable through the UI** and requires **no changes to the core BubbleLabs codebase**.

## 🚀 Features

### ✅ Zero Core Modifications
- **Standalone plugin** that works with any BubbleLabs installation
- **No changes** to core BubbleLabs codebase required
- **Clean separation** of concerns with well-defined interfaces

### 🎛️ Fully Configurable UI
- **Configuration panel** for easy setup and management
- **Real-time status monitoring** with visual indicators
- **Search options** with intelligent filtering
- **Comprehensive logging** and operation history

### 🔍 Semantic Search
- **Vector-based semantic search** for conceptual matching
- **Hybrid search** combining semantic and keyword approaches
- **Reranking** for improved result relevance
- **Advanced filtering** by document type, stage, team, and tags

### 📊 Knowledge Management
- **Automatic indexing** of workflow artifacts
- **Document ingestion** with rich metadata support
- **Batch operations** for efficient processing
- **Index statistics** and monitoring

### ⚡ Performance
- **Caching system** for improved response times
- **Configurable timeouts** and batch sizes
- **Operation history** tracking
- **Performance statistics** and metrics

## 📦 Installation

### Using npm

```bash
npm install bubblelabs-ragbits-plugin
```

### Using yarn

```bash
yarn add bubblelabs-ragbits-plugin
```

### Using pnpm

```bash
pnpm add bubblelabs-ragbits-plugin
```

## 🔧 Configuration

### Basic Setup

```typescript
import { RAGBitsPlugin, createPlugin } from 'bubblelabs-ragbits-plugin';

// Create plugin instance
const plugin = createPlugin({
  serverUrl: 'http://localhost:3000/ragbits',
  apiKey: 'your-api-key', // optional
  defaultTopK: 10
});

// Initialize the plugin
await plugin.initialize();
```

### Advanced Configuration

```typescript
import { DEFAULT_RAGBITS_CONFIG } from 'bubblelabs-ragbits-plugin';

const customConfig = {
  ...DEFAULT_RAGBITS_CONFIG,
  serverUrl: 'https://enterprise-ragbits.example.com',
  apiKey: process.env.RAGBITS_API_KEY,
  timeout: 60,
  defaultTopK: 20,
  defaultScoreThreshold: 0.8,
  enableHybridSearch: true,
  enableReranking: true,
  autoIndexArtifacts: true,
  indexingBatchSize: 200,
  enableCaching: true,
  cacheTTLSeconds: 7200 // 2 hours
};

const plugin = createPlugin(customConfig);
await plugin.initialize();
```

## 🎯 Usage

### Searching Documents

```typescript
import { useRAGBitsSearch } from 'bubblelabs-ragbits-plugin';

function MyComponent() {
  const search = useRAGBitsSearch();

  const handleSearch = async () => {
    const results = await search({
      query: 'microservices authentication',
      topK: 10,
      scoreThreshold: 0.7,
      filter: {
        documentType: 'solution',
        stage: 'stage_3'
      },
      enableHybridSearch: true,
      enableReranking: true
    });

    console.log('Found', results.totalResults, 'documents');
    results.results.forEach(result => {
      console.log('Score:', result.relevanceScore);
      console.log('Content:', result.content);
    });
  };

  return <button onClick={handleSearch}>Search</button>;
}
```

### Ingesting Documents

```typescript
import { useRAGBitsIngest } from 'bubblelabs-ragbits-plugin';

function IngestComponent() {
  const ingest = useRAGBitsIngest();

  const handleIngest = async () => {
    const response = await ingest({
      content: 'Microservices architecture with load balancing...',
      metadata: {
        documentType: 'solution',
        source: 'team-blue',
        stage: 'stage_3',
        team: 'blue',
        tags: ['microservices', 'architecture', 'scalability']
      }
    });

    console.log('Document ID:', response.documentId);
  };

  return <button onClick={handleIngest}>Ingest Document</button>;
}
```

### Using UI Components

```typescript
import { RAGBitsSearchPanel, RAGBitsSearchResults } from 'bubblelabs-ragbits-plugin';

function SearchInterface() {
  const [results, setResults] = useState([]);

  return (
    <div>
      <RAGBitsSearchPanel
        onResult={setResults}
        initialQuery="authentication system"
        showDebug={false}
      />
      <RAGBitsSearchResults
        results={results}
        onResultClick={(result) => console.log('Clicked:', result)}
        showMetadata={true}
        showScores={true}
      />
    </div>
  );
}
```

### Configuration Panel

```typescript
import { RAGBitsConfigPanel } from 'bubblelabs-ragbits-plugin';

function Settings() {
  const handleSave = (config) => {
    console.log('Saving config:', config);
    // Save configuration
  };

  return (
    <RAGBitsConfigPanel
      onSave={handleSave}
      onCancel={() => console.log('Cancelled')}
      initialConfig={{ serverUrl: 'http://localhost:3000/ragbits' }}
      showAdvanced={true}
    />
  );
}
```

## 🏗️ Architecture

### Plugin Structure

```
bubblelabs-ragbits-plugin/
├── src/
│   ├── types/
│   │   └── plugin-types.ts       # Type definitions
│   ├── lib/
│   │   └── ragbitsClient.ts      # HTTP client
│   ├── services/
│   │   └── ragbitsService.ts     # High-level service
│   ├── components/
│   │   ├── RAGBitsConfigPanel.tsx
│   │   ├── RAGBitsSearchPanel.tsx
│   │   ├── RAGBitsIngestPanel.tsx
│   │   ├── RAGBitsStatusIndicator.tsx
│   │   └── RAGBitsSearchResults.tsx
│   ├── hooks/
│   │   ├── useRAGBitsConfig.ts
│   │   ├── useRAGBitsState.ts
│   │   ├── useRAGBitsSearch.ts
│   │   └── useRAGBitsIngest.ts
│   ├── utils/
│   │   └── createRAGBitsPlugin.ts
│   └── index.ts
├── package.json
├── tsconfig.json
└── README.md
```

### Data Flow

```
BubbleLabs UI
    ↓
RAGBits Plugin Components
    ↓
Plugin Factory (createRAGBitsPlugin)
    ↓
RAGBits Service (with caching)
    ↓
RAGBits Client (HTTP)
    ↓
RAGBits Server
```

## 🔌 Integration with BubbleLabs

### Adding to BubbleLabs Integrations

Update `BubbleLab/apps/bubble-studio/src/lib/integrations.ts`:

```typescript
export const SERVICE_LOGOS: Readonly<Record<string, string>> = Object.freeze({
  // ... existing integrations
  RAGBits: '/integrations/ragbits.svg',
});

export const OPENEVOLVE_INTEGRATIONS: IntegrationLogo[] = [
  // ... existing integrations
  { name: 'RAGBits', file: SERVICE_LOGOS['RAGBits'] },
];
```

### Using in BubbleLabs Workflow

```typescript
import { RAGBitsPlugin } from 'bubblelabs-ragbits-plugin';

// In your workflow component
const plugin = RAGBitsPlugin;

// Search for relevant solutions
const similarSolutions = await plugin.search({
  query: currentProblem,
  filter: { documentType: 'solution' }
});

// Auto-index artifacts
if (config.autoIndexArtifacts) {
  await plugin.ingest({
    content: generatedSolution,
    metadata: {
      documentType: 'solution',
      stage: currentStage,
      team: currentTeam
    }
  });
}
```

## 📊 Plugin API

### Main Methods

- `initialize(config?)` - Initialize the plugin
- `search(request)` - Search for documents
- `ingest(request)` - Ingest a document
- `batchIngest(requests)` - Batch ingest documents
- `getIndexStats()` - Get index statistics
- `clearCache()` - Clear the cache
- `updateConfig(config)` - Update configuration
- `getStatistics()` - Get usage statistics
- `getStatus()` - Get plugin status

### Components

- `RAGBitsConfigPanel` - Configuration UI
- `RAGBitsSearchPanel` - Search interface
- `RAGBitsIngestPanel` - Document ingestion UI
- `RAGBitsStatusIndicator` - Status display
- `RAGBitsSearchResults` - Results display

### Hooks

- `useRAGBitsConfig()` - Configuration management
- `useRAGBitsState()` - Plugin state
- `useRAGBitsSearch()` - Search function
- `useRAGBitsIngest()` - Ingest function

## 🎨 Customization

### Custom Search Types

```typescript
import { RAGBITS_SEARCH_TYPES } from 'bubblelabs-ragbits-plugin';

// Use different search types
const semantic = await plugin.search({
  query: 'machine learning',
  enableHybridSearch: false // Semantic only
});

const hybrid = await plugin.search({
  query: 'machine learning',
  enableHybridSearch: true // Hybrid search
});
```

### Custom Filters

```typescript
const results = await plugin.search({
  query: 'authentication',
  filter: {
    documentType: 'solution',
    stage: 'stage_3',
    team: 'blue',
    tags: ['security', 'api']
  }
});
```

## 🐛 Troubleshooting

### Connection Issues

```typescript
// Test connection
const plugin = createPlugin({ serverUrl: 'http://localhost:3000/ragbits' });

try {
  await plugin.initialize();
  console.log('Connected successfully');
} catch (error) {
  console.error('Connection failed:', error);
}
```

### Debug Mode

```typescript
// Enable debug info
<RAGBitsSearchPanel
  onResult={handleResult}
  showDebug={true}
/>
```

### Checking Statistics

```typescript
const stats = plugin.getStatistics();
console.log('Total searches:', stats.totalSearches);
console.log('Success rate:', stats.successfulSearches / stats.totalSearches);
console.log('Average time:', stats.averageSearchTime);
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please read CONTRIBUTING.md for details.

## 📧 Support

For issues and questions:
- GitHub: https://github.com/openevolve/bubblelabs-ragbits-plugin
- Website: https://openevolve.com

## 🔗 Links

- [BubbleLabs](https://bubblelabs.com)
- [RAGBits](https://ragbits.com)
- [OpenEvolve](https://openevolve.com)

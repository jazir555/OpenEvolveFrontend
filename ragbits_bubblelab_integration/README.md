# Ragbits + BubbleLab Integration

This package provides integration between the Ragbits RAG framework and the BubbleLab workflow builder, enabling users to create, configure, and monitor RAG workflows through a visual interface.

## Features

- **Visual RAG Workflow Builder**: Create RAG workflows using BubbleLab's visual interface
- **Ragbits Integration**: Leverage Ragbits' document processing and search capabilities
- **Real-time Monitoring**: Monitor and debug RAG workflows in real-time
- **Configuration Management**: Map visual workflows to Ragbits configurations
- **Deployment Ready**: Export workflows as production-ready TypeScript code

## Installation

```bash
npm install @openevolve/ragbits-bubblelab-integration
```

## Usage

### Basic Setup

```typescript
import { RagbitsBubbleLabIntegration } from '@openevolve/ragbits-bubblelab-integration';

const integration = RagbitsBubbleLabIntegration.getInstance();

// Create a workflow engine
const workflowEngine = integration.createWorkflowEngine(bubbleLabConfig);

// Initialize the engine
await workflowEngine.initialize();

// Execute the workflow
const results = await workflowEngine.executeWorkflow();
```

### Creating RAG Bubbles

```typescript
import { 
  RAGBitsIngestBubble, 
  RAGBitsSearchBubble, 
  RAGBitsGenerationBubble 
} from '@openevolve/ragbits-bubblelab-integration';

// Create an ingest bubble
const ingestBubble = new RAGBitsIngestBubble({
  sourceType: 'file',
  sourcePath: './documents/',
  chunkSize: 1000,
  chunkOverlap: 200
});

// Create a search bubble
const searchBubble = new RAGBitsSearchBubble({
  topK: 5,
  scoreThreshold: 0.7,
  enableHybridSearch: true
});

// Create a generation bubble
const generationBubble = new RAGBitsGenerationBubble({
  llmModel: 'gpt-4o',
  temperature: 0.7,
  maxTokens: 1000
});
```

### Configuration Mapping

```typescript
import { ConfigMapper } from '@openevolve/ragbits-bubblelab-integration';

// Map a BubbleLab workflow to Ragbits configuration
const ragbitsConfig = ConfigMapper.mapBubbleLabToRagbits(bubbleLabConfig);

// Validate the configuration
const validation = ConfigMapper.validateBubbleLabConfig(bubbleLabConfig);
if (!validation.isValid) {
  console.error('Invalid configuration:', validation.errors);
}
```

### Monitoring and Debugging

```typescript
import { MonitoringService } from '@openevolve/ragbits-bubblelab-integration';

// Create a monitoring service
const monitoringService = integration.createMonitoringService({
  enableRealTimeMonitoring: true,
  enablePerformanceTracking: true,
  enableErrorTracking: true
});

// Listen for events
monitoringService.addEventListener((event) => {
  console.log('Monitoring event:', event);
});

// Get performance metrics
const metrics = monitoringService.getPerformanceMetrics();
console.log('Performance metrics:', metrics);
```

### Generating Configurations

```typescript
import { ConfigGenerator } from '@openevolve/ragbits-bubblelab-integration';

// Generate a Ragbits configuration from a BubbleLab workflow
const generatedConfig = integration.generateConfig(bubbleLabConfig, {
  validate: true,
  format: 'typescript',
  targetEnvironment: 'production'
});

console.log('Generated Ragbits config:', generatedConfig.ragbitsConfig);
```

## Components

### RAG Bubbles

- **RAGBitsIngestBubble**: Handles document ingestion into the RAG system
- **RAGBitsSearchBubble**: Performs semantic search on indexed documents
- **RAGBitsGenerationBubble**: Generates responses using retrieved context
- **RAGBitsIndexBubble**: Manages the vector index

### Services

- **ConfigMapper**: Maps BubbleLab workflows to Ragbits configurations
- **ConfigGenerator**: Generates Ragbits configurations from BubbleLab workflows
- **RAGBitsWorkflowEngine**: Executes RAG workflows
- **RagbitsProcessorIntegration**: Integrates with the Ragbits document processor
- **MonitoringService**: Provides real-time monitoring and debugging

## Architecture

The integration follows a modular architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BubbleLab UI  │◄──►│  Integration    │◄──►│   Ragbits Core  │
│   (Workflow     │    │     Layer       │    │   (RAG Engine)  │
│   Builder)      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  RAG Workflows  │    │  Configuration  │    │  RAG Components │
│  (Visual Flow)  │    │  Management     │    │  (Indexing,    │
│                 │    │                 │    │   Retrieval,    │
│                 │    │                 │    │   Generation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Configuration Options

### Workflow Engine Options

- `timeout`: Execution timeout in milliseconds (default: 30000)
- `maxRetries`: Maximum number of retries for failed nodes (default: 3)
- `enableLogging`: Enable logging (default: true)
- `logLevel`: Log level ('info', 'debug', 'warn', 'error') (default: 'info')

### Monitoring Configuration

- `enableRealTimeMonitoring`: Enable real-time monitoring (default: true)
- `enablePerformanceTracking`: Enable performance tracking (default: true)
- `enableErrorTracking`: Enable error tracking (default: true)
- `logLevel`: Log level for monitoring (default: 'info')
- `retentionPeriod`: Event retention period in days (default: 7)
- `samplingRate`: Sampling rate for events (0.0 to 1.0, default: 1.0)
- `enableAlerting`: Enable alerting (default: true)

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) for more information.

## License

MIT
# API Reference: Ragbits + BubbleLab Integration

Detailed reference for all classes, methods, and interfaces in the Ragbits + BubbleLab Integration.

## Classes

### RagbitsBubbleLabIntegration

Main integration class that provides access to all functionality.

#### Methods

##### getInstance()
Returns the singleton instance of the integration.

```typescript
static getInstance(): RagbitsBubbleLabIntegration
```

**Returns:** `RagbitsBubbleLabIntegration` - The singleton instance

##### createWorkflowEngine(workflowConfig, options?)
Creates a new workflow engine instance.

```typescript
createWorkflowEngine(
  workflowConfig: BubbleLabWorkflowConfig,
  options?: WorkflowExecutionOptions
): RAGBitsWorkflowEngine
```

**Parameters:**
- `workflowConfig`: `BubbleLabWorkflowConfig` - Configuration for the workflow
- `options?`: `WorkflowExecutionOptions` - Optional execution options

**Returns:** `RAGBitsWorkflowEngine` - New workflow engine instance

##### createProcessorIntegration(config?)
Creates a new processor integration instance.

```typescript
createProcessorIntegration(
  config?: ProcessorIntegrationConfig
): RagbitsProcessorIntegration
```

**Parameters:**
- `config?`: `ProcessorIntegrationConfig` - Optional configuration

**Returns:** `RagbitsProcessorIntegration` - New processor integration instance

##### createMonitoringService(config?)
Creates a new monitoring service instance.

```typescript
createMonitoringService(
  config?: Partial<MonitoringConfig>
): MonitoringService
```

**Parameters:**
- `config?`: `Partial<MonitoringConfig>` - Optional monitoring configuration

**Returns:** `MonitoringService` - New monitoring service instance

##### generateConfig(bubbleLabConfig, options?)
Generates a Ragbits configuration from a BubbleLab workflow.

```typescript
generateConfig(
  bubbleLabConfig: BubbleLabWorkflowConfig,
  options?: GenerationOptions
): GeneratedConfig
```

**Parameters:**
- `bubbleLabConfig`: `BubbleLabWorkflowConfig` - BubbleLab workflow configuration
- `options?`: `GenerationOptions` - Optional generation options

**Returns:** `GeneratedConfig` - Generated Ragbits configuration

##### mapConfig(bubbleLabConfig)
Maps a BubbleLab workflow to a Ragbits configuration.

```typescript
mapConfig(bubbleLabConfig: BubbleLabWorkflowConfig): RagbitsConfig
```

**Parameters:**
- `bubbleLabConfig`: `BubbleLabWorkflowConfig` - BubbleLab workflow configuration

**Returns:** `RagbitsConfig` - Mapped Ragbits configuration

---

### RAGBitsWorkflowEngine

Executes RAG workflows defined in BubbleLab.

#### Constructor

```typescript
constructor(
  workflowConfig: BubbleLabWorkflowConfig,
  options?: WorkflowExecutionOptions
)
```

**Parameters:**
- `workflowConfig`: `BubbleLabWorkflowConfig` - Configuration for the workflow
- `options?`: `WorkflowExecutionOptions` - Optional execution options

#### Methods

##### initialize()
Initializes the workflow engine by creating instances of all nodes.

```typescript
async initialize(): Promise<void>
```

##### executeWorkflow(initialInput?)
Executes the entire workflow with optional initial input.

```typescript
async executeWorkflow(
  initialInput?: any
): Promise<WorkflowExecutionResult[]>
```

**Parameters:**
- `initialInput?`: `any` - Optional initial input for the workflow

**Returns:** `Promise<WorkflowExecutionResult[]>` - Results of workflow execution

##### getExecutionHistory()
Returns the execution history of the workflow.

```typescript
getExecutionHistory(): WorkflowExecutionResult[]
```

**Returns:** `WorkflowExecutionResult[]` - Execution history

##### getNodeResult(nodeId)
Gets the result of a specific node.

```typescript
getNodeResult(nodeId: string): WorkflowExecutionResult | undefined
```

**Parameters:**
- `nodeId`: `string` - ID of the node

**Returns:** `WorkflowExecutionResult | undefined` - Node result or undefined

##### reset()
Resets the execution history.

```typescript
reset(): void
```

##### dispose()
Disposes of resources used by the engine.

```typescript
async dispose(): Promise<void>
```

---

### RagbitsProcessorIntegration

Integrates with the Ragbits document processor.

#### Constructor

```typescript
constructor(config?: ProcessorIntegrationConfig)
```

**Parameters:**
- `config?`: `ProcessorIntegrationConfig` - Optional configuration

#### Methods

##### initialize()
Initializes the processor integration.

```typescript
async initialize(): Promise<void>
```

##### processDocument(source, content, metadata?)
Processes a single document.

```typescript
async processDocument(
  source: string,
  content: string,
  metadata?: Record<string, any>
): Promise<ProcessedDocument>
```

**Parameters:**
- `source`: `string` - Source identifier for the document
- `content`: `string` - Content of the document
- `metadata?`: `Record<string, any>` - Optional metadata

**Returns:** `Promise<ProcessedDocument>` - Processed document result

##### addDocument(source, content, metadata?)
Adds a document to the processing queue.

```typescript
async addDocument(
  source: string,
  content: string,
  metadata?: Record<string, any>
): Promise<ProcessedDocument>
```

**Parameters:**
- `source`: `string` - Source identifier for the document
- `content`: `string` - Content of the document
- `metadata?`: `Record<string, any>` - Optional metadata

**Returns:** `Promise<ProcessedDocument>` - Processed document result

##### search(query, topK?, filters?, minScore?)
Searches for documents using the Ragbits processor.

```typescript
async search(
  query: string,
  topK?: number,
  filters?: Record<string, any>,
  minScore?: number
): Promise<Array<{
  documentId: string;
  content: string;
  metadata: Record<string, any>;
  score: number;
}>>
```

**Parameters:**
- `query`: `string` - Search query
- `topK?`: `number` - Number of results to return (default: 5)
- `filters?`: `Record<string, any>` - Optional filters
- `minScore?`: `number` - Minimum score threshold (default: 0.0)

**Returns:** `Promise<Array<{documentId, content, metadata, score}>>` - Search results

##### getStats()
Returns statistics about document processing.

```typescript
getStats(): ProcessingStats
```

**Returns:** `ProcessingStats` - Processing statistics

##### getQueueSize()
Returns the number of documents in the processing queue.

```typescript
getQueueSize(): number
```

**Returns:** `number` - Queue size

##### clearQueue()
Clears the processing queue.

```typescript
clearQueue(): void
```

##### clearStore()
Clears the vector store.

```typescript
async clearStore(): Promise<boolean>
```

**Returns:** `Promise<boolean>` - True if successful, false otherwise

##### getIndexStats()
Returns index statistics from the processor.

```typescript
async getIndexStats(): Promise<any>
```

**Returns:** `Promise<any>` - Index statistics

##### dispose()
Disposes of resources used by the processor integration.

```typescript
async dispose(): Promise<void>
```

---

### MonitoringService

Provides real-time monitoring and debugging capabilities.

#### Constructor

```typescript
constructor(config?: Partial<MonitoringConfig>)
```

**Parameters:**
- `config?`: `Partial<MonitoringConfig>` - Optional monitoring configuration

#### Methods

##### addEventListener(listener)
Registers an event listener.

```typescript
addEventListener(listener: (event: MonitoringEvent) => void): void
```

**Parameters:**
- `listener`: `(event: MonitoringEvent) => void` - Event listener function

##### removeEventListener(listener)
Removes an event listener.

```typescript
removeEventListener(listener: (event: MonitoringEvent) => void): void
```

**Parameters:**
- `listener`: `(event: MonitoringEvent) => void` - Event listener function to remove

##### addAlertCallback(callback)
Registers an alert callback.

```typescript
addAlertCallback(callback: (alert: string) => void): void
```

**Parameters:**
- `callback`: `(alert: string) => void` - Alert callback function

##### removeAlertCallback(callback)
Removes an alert callback.

```typescript
removeAlertCallback(callback: (alert: string) => void): void
```

**Parameters:**
- `callback`: `(alert: string) => void` - Alert callback function to remove

##### logWorkflowStart(workflowId, metadata?)
Logs a workflow start event.

```typescript
logWorkflowStart(workflowId: string, metadata?: Record<string, any>): void
```

**Parameters:**
- `workflowId`: `string` - ID of the workflow
- `metadata?`: `Record<string, any>` - Optional metadata

##### logWorkflowComplete(workflowId, duration, metadata?)
Logs a workflow completion event.

```typescript
logWorkflowComplete(
  workflowId: string,
  duration: number,
  metadata?: Record<string, any>
): void
```

**Parameters:**
- `workflowId`: `string` - ID of the workflow
- `duration`: `number` - Duration of execution in milliseconds
- `metadata?`: `Record<string, any>` - Optional metadata

##### logWorkflowError(workflowId, error, metadata?)
Logs a workflow error event.

```typescript
logWorkflowError(
  workflowId: string,
  error: string,
  metadata?: Record<string, any>
): void
```

**Parameters:**
- `workflowId`: `string` - ID of the workflow
- `error`: `string` - Error message
- `metadata?`: `Record<string, any>` - Optional metadata

##### logNodeStart(workflowId, nodeId, metadata?)
Logs a node start event.

```typescript
logNodeStart(
  workflowId: string,
  nodeId: string,
  metadata?: Record<string, any>
): void
```

**Parameters:**
- `workflowId`: `string` - ID of the workflow
- `nodeId`: `string` - ID of the node
- `metadata?`: `Record<string, any>` - Optional metadata

##### logNodeComplete(workflowId, nodeId, duration, metadata?)
Logs a node completion event.

```typescript
logNodeComplete(
  workflowId: string,
  nodeId: string,
  duration: number,
  metadata?: Record<string, any>
): void
```

**Parameters:**
- `workflowId`: `string` - ID of the workflow
- `nodeId`: `string` - ID of the node
- `duration`: `number` - Duration of execution in milliseconds
- `metadata?`: `Record<string, any>` - Optional metadata

##### logNodeError(workflowId, nodeId, error, metadata?)
Logs a node error event.

```typescript
logNodeError(
  workflowId: string,
  nodeId: string,
  error: string,
  metadata?: Record<string, any>
): void
```

**Parameters:**
- `workflowId`: `string` - ID of the workflow
- `nodeId`: `string` - ID of the node
- `error`: `string` - Error message
- `metadata?`: `Record<string, any>` - Optional metadata

##### recordDebugInfo(nodeId, debugInfo)
Records debug information for a node.

```typescript
recordDebugInfo(nodeId: string, debugInfo: Omit<DebugInfo, 'nodeId'>): void
```

**Parameters:**
- `nodeId`: `string` - ID of the node
- `debugInfo`: `Omit<DebugInfo, 'nodeId'>` - Debug information

##### getEventLog()
Returns the current event log.

```typescript
getEventLog(): MonitoringEvent[]
```

**Returns:** `MonitoringEvent[]` - Event log

##### getWorkflowEvents(workflowId)
Returns events for a specific workflow.

```typescript
getWorkflowEvents(workflowId: string): MonitoringEvent[]
```

**Parameters:**
- `workflowId`: `string` - ID of the workflow

**Returns:** `MonitoringEvent[]` - Workflow events

##### getNodeEvents(nodeId)
Returns events for a specific node.

```typescript
getNodeEvents(nodeId: string): MonitoringEvent[]
```

**Parameters:**
- `nodeId`: `string` - ID of the node

**Returns:** `MonitoringEvent[]` - Node events

##### getPerformanceMetrics()
Returns performance metrics.

```typescript
getPerformanceMetrics(): PerformanceMetrics
```

**Returns:** `PerformanceMetrics` - Performance metrics

##### getDebugInfo(nodeId)
Returns debug information for a node.

```typescript
getDebugInfo(nodeId: string): DebugInfo[]
```

**Parameters:**
- `nodeId`: `string` - ID of the node

**Returns:** `DebugInfo[]` - Debug information

##### getAllDebugInfo()
Returns all debug information.

```typescript
getAllDebugInfo(): Map<string, DebugInfo[]>
```

**Returns:** `Map<string, DebugInfo[]>` - All debug information

##### getActiveWorkflows()
Returns active workflows.

```typescript
getActiveWorkflows(): string[]
```

**Returns:** `string[]` - Active workflow IDs

##### getWorkflowStats()
Returns workflow statistics.

```typescript
getWorkflowStats(): {
  totalWorkflows: number;
  activeWorkflows: number;
  completedWorkflows: number;
  errorWorkflows: number;
  averageExecutionTime: number;
}
```

**Returns:** Object with workflow statistics

##### exportData(format?)
Exports monitoring data.

```typescript
exportData(format?: 'json' | 'csv'): string
```

**Parameters:**
- `format?`: `'json' | 'csv'` - Export format (default: 'json')

**Returns:** `string` - Exported data

##### reset()
Resets the monitoring service.

```typescript
reset(): void
```

---

## Interfaces

### BubbleLabWorkflowConfig

Configuration for a BubbleLab workflow.

```typescript
interface BubbleLabWorkflowConfig {
  id: string;
  name: string;
  description: string;
  nodes: BubbleLabNode[];
  edges: BubbleLabEdge[];
  metadata: Record<string, any>;
}
```

### BubbleLabNode

Definition of a node in a BubbleLab workflow.

```typescript
interface BubbleLabNode {
  id: string;
  type: string; // 'ragbits-ingest', 'ragbits-search', 'ragbits-generation', 'ragbits-index'
  position: { x: number; y: number };
  data: Record<string, any>;
}
```

### BubbleLabEdge

Definition of an edge connecting nodes in a BubbleLab workflow.

```typescript
interface BubbleLabEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}
```

### WorkflowExecutionOptions

Options for workflow execution.

```typescript
interface WorkflowExecutionOptions {
  timeout?: number; // in milliseconds
  maxRetries?: number;
  enableLogging?: boolean;
  logLevel?: 'info' | 'debug' | 'warn' | 'error';
}
```

### ProcessorIntegrationConfig

Configuration for processor integration.

```typescript
interface ProcessorIntegrationConfig {
  enableAutoIndexing?: boolean;
  autoIndexInterval?: number; // in seconds
  batchSize?: number;
  enableCaching?: boolean;
  cacheTTL?: number; // in seconds
  enableMonitoring?: boolean;
  maxConcurrentProcesses?: number;
}
```

### MonitoringConfig

Configuration for monitoring service.

```typescript
interface MonitoringConfig {
  enableRealTimeMonitoring: boolean;
  enablePerformanceTracking: boolean;
  enableErrorTracking: boolean;
  enableTokenTracking: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  retentionPeriod: number; // in days
  samplingRate: number; // 0.0 to 1.0
  enableAlerting: boolean;
  alertThresholds: {
    executionTime: number; // in ms
    errorRate: number; // percentage
    memoryUsage: number; // in MB
  };
}
```

### GenerationOptions

Options for configuration generation.

```typescript
interface GenerationOptions {
  includeComments?: boolean;
  format?: 'json' | 'yaml' | 'typescript';
  validate?: boolean;
  generateDeploymentFiles?: boolean;
  targetEnvironment?: 'development' | 'staging' | 'production';
}
```

### MonitoringEvent

Event in the monitoring system.

```typescript
interface MonitoringEvent {
  id: string;
  timestamp: Date;
  eventType: 'node_start' | 'node_complete' | 'node_error' | 'workflow_start' | 'workflow_complete' | 'workflow_error';
  workflowId: string;
  nodeId?: string;
  duration?: number;
  metadata?: Record<string, any>;
  error?: string;
}
```

### PerformanceMetrics

Performance metrics for monitoring.

```typescript
interface PerformanceMetrics {
  workflowExecutionTime: number;
  nodeExecutionTimes: Record<string, number>;
  tokensUsed: number;
  memoryUsage: number;
  cacheHitRate: number;
  errorRate: number;
  throughput: number; // operations per minute
}
```

### DebugInfo

Debug information for a node.

```typescript
interface DebugInfo {
  nodeId: string;
  inputs: any;
  outputs: any;
  executionTime: number;
  logs: string[];
  error?: string;
}
```

### GeneratedConfig

Generated configuration result.

```typescript
interface GeneratedConfig {
  ragbitsConfig: RagbitsConfig;
  deploymentManifest?: any;
  environmentConfig?: any;
  validationErrors?: string[];
}
```

### RagbitsConfig

Configuration for Ragbits system.

```typescript
interface RagbitsConfig {
  documentProcessor: {
    embedding_model: string;
    vector_store_type: 'memory' | 'qdrant';
    qdrant_url?: string;
    qdrant_collection?: string;
    chunk_size: number;
    chunk_overlap: number;
    min_chunk_size: number;
  };
  search: {
    default_top_k: number;
    default_score_threshold: number;
    enable_hybrid_search: boolean;
    enable_reranking: boolean;
  };
  generation: {
    default_model: string;
    default_temperature: number;
    default_max_tokens: number;
  };
  workflow: {
    name: string;
    description: string;
    nodes: RagbitsNodeConfig[];
    connections: RagbitsConnection[];
  };
}
```

### RagbitsNodeConfig

Configuration for a Ragbits node.

```typescript
interface RagbitsNodeConfig {
  id: string;
  type: 'ingest' | 'search' | 'generation' | 'index';
  config: any;
  inputs?: string[];
  outputs?: string[];
}
```

### RagbitsConnection

Connection between Ragbits nodes.

```typescript
interface RagbitsConnection {
  sourceNodeId: string;
  sourceOutput: string;
  targetNodeId: string;
  targetInput: string;
}
```

### ProcessedDocument

Result of document processing.

```typescript
interface ProcessedDocument {
  id: string;
  source: string;
  content: string;
  metadata: Record<string, any>;
  processingTime: number;
  success: boolean;
  error?: string;
}
```

### ProcessingStats

Statistics about document processing.

```typescript
interface ProcessingStats {
  totalProcessed: number;
  successful: number;
  failed: number;
  averageProcessingTime: number;
  lastProcessed: Date;
  queueSize: number;
}
```

### WorkflowExecutionResult

Result of workflow execution.

```typescript
interface WorkflowExecutionResult {
  success: boolean;
  nodeId: string;
  output: any;
  executionTime: number;
  error?: string;
}
```

## Bubble Components

### BaseBubble

Abstract base class for all bubble components.

#### Constructor

```typescript
constructor(config: ConfigType)
```

**Parameters:**
- `config`: `ConfigType` - Configuration for the bubble

#### Methods

##### initialize()
Initializes the bubble.

```typescript
async initialize(): Promise<void>
```

##### action(input)
Performs the bubble's action.

```typescript
abstract action(input: InputType): Promise<OutputType>
```

**Parameters:**
- `input`: `InputType` - Input for the action

**Returns:** `Promise<OutputType>` - Output of the action

##### dispose()
Disposes of resources used by the bubble.

```typescript
async dispose(): Promise<void>
```

### RAGBitsIngestBubble

Handles document ingestion into the RAG system.

#### Constructor

```typescript
constructor(config: RAGBitsIngestConfig)
```

**Parameters:**
- `config`: `RAGBitsIngestConfig` - Configuration for the ingest bubble

### RAGBitsSearchBubble

Performs semantic search on indexed documents.

#### Constructor

```typescript
constructor(config: RAGBitsSearchConfig)
```

**Parameters:**
- `config`: `RAGBitsSearchConfig` - Configuration for the search bubble

### RAGBitsGenerationBubble

Generates responses using retrieved context.

#### Constructor

```typescript
constructor(config: RAGBitsGenerationConfig)
```

**Parameters:**
- `config`: `RAGBitsGenerationConfig` - Configuration for the generation bubble

### RAGBitsIndexBubble

Manages the vector index for document storage.

#### Constructor

```typescript
constructor(config: RAGBitsIndexConfig)
```

**Parameters:**
- `config`: `RAGBitsIndexConfig` - Configuration for the index bubble
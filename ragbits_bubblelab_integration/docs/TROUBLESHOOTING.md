# Troubleshooting Guide: Ragbits + BubbleLab Integration

This guide provides solutions to common issues you may encounter when using the Ragbits + BubbleLab Integration.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Workflow Execution Issues](#workflow-execution-issues)
- [Performance Problems](#performance-problems)
- [Monitoring Issues](#monitoring-issues)
- [Error Handling](#error-handling)
- [Common Scenarios](#common-scenarios)

## Installation Issues

### Problem: Cannot install the package
**Error:** `npm install @openevolve/ragbits-bubblelab-integration` fails

**Solutions:**
1. Check Node.js version: Ensure you're using Node.js 16 or higher
   ```bash
   node --version
   ```
2. Clear npm cache:
   ```bash
   npm cache clean --force
   ```
3. Try installing with verbose logging:
   ```bash
   npm install @openevolve/ragbits-bubblelab-integration --verbose
   ```

### Problem: Module not found after installation
**Error:** `Cannot find module '@openevolve/ragbits-bubblelab-integration'`

**Solutions:**
1. Verify installation:
   ```bash
   npm list @openevolve/ragbits-bubblelab-integration
   ```
2. Check your import statement:
   ```typescript
   // Correct
   import { RagbitsBubbleLabIntegration } from '@openevolve/ragbits-bubblelab-integration';
   
   // Incorrect
   import { RagbitsBubbleLabIntegration } from 'ragbits-bubblelab-integration';
   ```
3. If using TypeScript, ensure your `tsconfig.json` has proper module resolution:
   ```json
   {
     "compilerOptions": {
       "moduleResolution": "node",
       "esModuleInterop": true
     }
   }
   ```

## Configuration Problems

### Problem: Invalid workflow configuration
**Error:** `Invalid workflow configuration: Node missing ID`

**Solutions:**
1. Ensure all nodes have unique IDs:
   ```typescript
   const workflow = {
     nodes: [
       {
         id: 'unique-node-id', // Must be unique and non-empty
         type: 'ragbits-ingest',
         // ... other properties
       }
     ]
   };
   ```

2. Validate your workflow configuration:
   ```typescript
   import { ConfigMapper } from '@openevolve/ragbits-bubblelab-integration';
   
   const validation = ConfigMapper.validateBubbleLabConfig(workflowConfig);
   if (!validation.isValid) {
     console.error('Validation errors:', validation.errors);
   }
   ```

### Problem: Unknown node type
**Error:** `Unknown node type: ragbits-unknown`

**Solutions:**
1. Check supported node types:
   - `ragbits-ingest` - Document ingestion
   - `ragbits-search` - Semantic search
   - `ragbits-generation` - Response generation
   - `ragbits-index` - Index management

2. Verify the type property:
   ```typescript
   // Correct
   {
     id: 'search-node',
     type: 'ragbits-search', // Use correct type
     // ...
   }
   
   // Incorrect
   {
     id: 'search-node',
     type: 'ragbits-search-node', // Wrong type
     // ...
   }
   ```

### Problem: Configuration validation fails
**Error:** Configuration validation errors during generation

**Solutions:**
1. Enable validation during config generation:
   ```typescript
   const config = integration.generateConfig(workflowConfig, {
     validate: true
   });
   
   if (config.validationErrors) {
     console.error('Validation errors:', config.validationErrors);
   }
   ```

2. Check required properties for each node type:
   - **Ingest nodes:** Require `sourceType` and `sourcePath`
   - **Search nodes:** May require `topK` for proper operation
   - **Generation nodes:** May require `llmModel`
   - **Index nodes:** May require `vectorStoreType`

## Workflow Execution Issues

### Problem: Workflow execution times out
**Error:** `Node [node-id] execution timed out`

**Solutions:**
1. Increase the timeout in workflow engine options:
   ```typescript
   const engine = integration.createWorkflowEngine(workflowConfig, {
     timeout: 60000, // 60 seconds instead of default 30
     maxRetries: 3
   });
   ```

2. For long-running operations, consider using a higher timeout:
   ```typescript
   const engine = integration.createWorkflowEngine(workflowConfig, {
     timeout: 120000, // 2 minutes for heavy processing
   });
   ```

### Problem: Node execution fails
**Error:** `Node [node-id] failed: [specific error message]`

**Solutions:**
1. Check the specific error message in the result:
   ```typescript
   const results = await engine.executeWorkflow(input);
   for (const result of results) {
     if (!result.success) {
       console.error(`Node ${result.nodeId} failed:`, result.error);
     }
   }
   ```

2. Enable detailed logging:
   ```typescript
   const engine = integration.createWorkflowEngine(workflowConfig, {
     enableLogging: true,
     logLevel: 'debug' // Use 'debug' for more details
   });
   ```

### Problem: Documents not ingesting properly
**Error:** Ingestion nodes report failures

**Solutions:**
1. Verify source path exists and is accessible:
   ```typescript
   {
     id: 'ingest-node',
     type: 'ragbits-ingest',
     data: {
       sourceType: 'file',
       sourcePath: './existing-directory/', // Ensure this path exists
       // ...
     }
   }
   ```

2. For text content, ensure content is provided:
   ```typescript
   // When calling the action directly
   const result = await ingestBubble.action({
     content: 'Actual content to ingest', // Don't pass empty content
     metadata: { /* optional metadata */ }
   });
   ```

### Problem: Search returns no results
**Error:** Search nodes return empty results

**Solutions:**
1. Check if documents were properly ingested first
2. Adjust search parameters:
   ```typescript
   {
     id: 'search-node',
     type: 'ragbits-search',
     data: {
       topK: 10, // Return more results
       scoreThreshold: 0.3, // Lower threshold for more lenient matching
       enableHybridSearch: true // Enable hybrid search
     }
   }
   ```

3. Verify the search query is appropriate for your documents

## Performance Problems

### Problem: Slow workflow execution
**Symptoms:** Workflows taking longer than expected to complete

**Solutions:**
1. Use Qdrant instead of memory for vector storage in production:
   ```typescript
   {
     id: 'index-node',
     type: 'ragbits-index',
     data: {
       vectorStoreType: 'qdrant', // Use Qdrant for better performance
       embeddingModel: 'text-embedding-3-small'
     }
   }
   ```

2. Optimize chunk sizes:
   ```typescript
   {
     id: 'ingest-node',
     type: 'ragbits-ingest',
     data: {
       chunkSize: 500, // Smaller chunks for better precision
       chunkOverlap: 50 // Appropriate overlap
     }
   }
   ```

3. Monitor performance metrics:
   ```typescript
   const monitoring = integration.createMonitoringService();
   const metrics = monitoring.getPerformanceMetrics();
   console.log('Execution times:', metrics.nodeExecutionTimes);
   ```

### Problem: High memory usage
**Symptoms:** Application consuming excessive memory

**Solutions:**
1. Use Qdrant for vector storage instead of memory:
   ```typescript
   vectorStoreType: 'qdrant' // Instead of 'memory'
   ```

2. Process documents in smaller batches:
   ```typescript
   const processor = integration.createProcessorIntegration({
     batchSize: 5, // Smaller batch size
     maxConcurrentProcesses: 3 // Limit concurrent processes
   });
   ```

3. Implement proper resource cleanup:
   ```typescript
   try {
     // ... workflow execution
   } finally {
     await engine.dispose(); // Always dispose of resources
   }
   ```

## Monitoring Issues

### Problem: Events not being logged
**Symptoms:** Monitoring service not capturing events

**Solutions:**
1. Ensure monitoring is enabled:
   ```typescript
   const monitoring = integration.createMonitoringService({
     enableRealTimeMonitoring: true,
     enablePerformanceTracking: true
   });
   ```

2. Add event listeners before executing workflows:
   ```typescript
   monitoring.addEventListener((event) => {
     console.log('Captured event:', event);
   });
   
   // Then execute workflow
   const results = await engine.executeWorkflow(input);
   ```

### Problem: Metrics not updating
**Symptoms:** Performance metrics showing 0 or incorrect values

**Solutions:**
1. Execute workflows before requesting metrics:
   ```typescript
   // Execute workflow first
   await engine.executeWorkflow(input);
   
   // Then get metrics
   const metrics = monitoring.getPerformanceMetrics();
   ```

2. Ensure monitoring service is attached to the workflow execution process

## Error Handling

### Problem: Unhandled promise rejection
**Error:** `UnhandledPromiseRejectionWarning`

**Solutions:**
1. Always use try/catch with async operations:
   ```typescript
   try {
     const results = await engine.executeWorkflow(input);
     // Handle results
   } catch (error) {
     console.error('Workflow execution failed:', error);
   }
   ```

2. Use Promise wrappers for error handling:
   ```typescript
   async function safeExecute<T>(fn: () => Promise<T>): Promise<T | null> {
     try {
       return await fn();
     } catch (error) {
       console.error('Execution failed:', error);
       return null;
     }
   }
   ```

### Problem: Silent failures
**Symptoms:** Operations appear to succeed but don't produce expected results

**Solutions:**
1. Always check result success flags:
   ```typescript
   const results = await engine.executeWorkflow(input);
   const failedResults = results.filter(r => !r.success);
   if (failedResults.length > 0) {
     console.warn('Some nodes failed:', failedResults);
   }
   ```

2. Enable logging to catch silent issues:
   ```typescript
   const engine = integration.createWorkflowEngine(workflowConfig, {
     enableLogging: true,
     logLevel: 'info'
   });
   ```

## Common Scenarios

### Scenario: Migrating from development to production
**Issue:** Configuration that works in development fails in production

**Solution:**
1. Use environment-specific configuration generation:
   ```typescript
   const config = integration.generateConfig(workflowConfig, {
     targetEnvironment: 'production' // Applies production optimizations
   });
   ```

2. Update to use Qdrant in production:
   ```typescript
   // Production config will automatically use Qdrant
   // instead of memory storage
   ```

### Scenario: Large document processing
**Issue:** Processing large documents or many documents at once

**Solution:**
1. Use the processor integration with queuing:
   ```typescript
   const processor = integration.createProcessorIntegration({
     batchSize: 10,
     maxConcurrentProcesses: 5,
     enableAutoIndexing: true
   });
   
   // Process documents one by one
   for (const doc of largeDocumentSet) {
     await processor.addDocument(doc.path, doc.content, doc.metadata);
   }
   ```

2. Monitor queue size and process in batches:
   ```typescript
   while (processor.getQueueSize() > 0) {
     await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 sec
   }
   ```

### Scenario: Custom node implementation
**Issue:** Want to extend functionality with custom nodes

**Solution:**
1. Extend the BaseBubble class:
   ```typescript
   import { BaseBubble } from '@openevolve/ragbits-bubblelab-integration';
   
   class CustomBubble extends BaseBubble<CustomConfig, CustomInput, CustomOutput> {
     async action(input: CustomInput): Promise<CustomOutput> {
       // Implement custom logic
       return { result: 'custom result' };
     }
   }
   ```

2. Register the custom node type in your workflow configuration

## Need More Help?

If you're still experiencing issues:

1. **Check the logs:** Enable debug logging to get more detailed information
2. **Review the API documentation:** Ensure you're using the correct parameters
3. **Run the examples:** Test with the provided example code to isolate the issue
4. **Community support:** Reach out to the community forums or issue tracker
5. **Create a minimal reproduction:** Create a minimal example that reproduces your issue
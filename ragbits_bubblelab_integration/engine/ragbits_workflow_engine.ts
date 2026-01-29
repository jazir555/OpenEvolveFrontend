/**
 * RAGBits Workflow Engine for BubbleLab
 *
 * This engine executes RAG workflows defined in BubbleLab using Ragbits components
 */

import { BaseBubble } from '../bubbles/BaseBubble';
import { ConfigMapper, BubbleLabWorkflowConfig, RagbitsNodeConfig, RagbitsConnection } from '../types';
import { RAGBitsIngestBubble } from '../bubbles/RAGBitsIngestBubble';
import { RAGBitsSearchBubble } from '../bubbles/RAGBitsSearchBubble';
import { RAGBitsGenerationBubble } from '../bubbles/RAGBitsGenerationBubble';
import { RAGBitsIndexBubble } from '../bubbles/RAGBitsIndexBubble';
import { WorkflowExecutionResult, WorkflowExecutionOptions } from '../types';
import { Logger, generateId } from '../utils/common.utils';

export class RAGBitsWorkflowEngine {
  private workflowConfig: BubbleLabWorkflowConfig;
  private nodeInstances: Map<string, BaseBubble<any, any, any>>;
  private executionHistory: WorkflowExecutionResult[];
  private options: WorkflowExecutionOptions;
  private logger: Logger;
  private executionId: string;

  constructor(workflowConfig: BubbleLabWorkflowConfig, options: WorkflowExecutionOptions = {}) {
    this.workflowConfig = workflowConfig;
    this.nodeInstances = new Map();
    this.executionHistory = [];
    this.executionId = generateId('workflow-execution');
    this.logger = new Logger({ level: 'info', prefix: `RAGBitsWorkflowEngine-${this.executionId}` });

    this.options = {
      timeout: 30000, // 30 seconds default
      maxRetries: 3,
      enableLogging: true,
      logLevel: 'info',
      ...options
    };
  }

  /**
   * Initializes the workflow engine by creating instances of all nodes
   */
  async initialize(): Promise<void> {
    this.logger.info(`Initializing workflow: ${this.workflowConfig.name} (ID: ${this.workflowConfig.id})`);

    // Validate the workflow configuration
    const validation = ConfigMapper.validateBubbleLabConfig(this.workflowConfig);
    if (!validation.isValid) {
      throw new Error(`Invalid workflow configuration: ${validation.errors.join(', ')}`);
    }

    // Create instances for each node
    for (const node of this.workflowConfig.nodes) {
      const nodeInstance = await this.createNodeInstance(node);
      this.nodeInstances.set(node.id, nodeInstance);
      this.logger.debug(`Created instance for node: ${node.id} (${node.type})`);
    }

    this.logger.info(`Workflow initialized with ${this.nodeInstances.size} nodes`);
  }

  /**
   * Creates an instance of a node based on its type
   */
  private async createNodeInstance(node: any): Promise<BaseBubble<any, any, any>> {
    switch (node.type) {
      case 'ragbits-ingest':
        const ingestConfig = this.mapToIngestConfig(node.data);
        const ingestBubble = new RAGBitsIngestBubble(ingestConfig);
        await ingestBubble.initialize();
        return ingestBubble;

      case 'ragbits-search':
        const searchConfig = this.mapToSearchConfig(node.data);
        const searchBubble = new RAGBitsSearchBubble(searchConfig);
        await searchBubble.initialize();
        return searchBubble;

      case 'ragbits-generation':
        const generationConfig = this.mapToGenerationConfig(node.data);
        const generationBubble = new RAGBitsGenerationBubble(generationConfig);
        await generationBubble.initialize();
        return generationBubble;

      case 'ragbits-index':
        const indexConfig = this.mapToIndexConfig(node.data);
        const indexBubble = new RAGBitsIndexBubble(indexConfig);
        await indexBubble.initialize();
        return indexBubble;

      default:
        throw new Error(`Unsupported node type: ${node.type}`);
    }
  }

  /**
   * Maps node data to RAGBitsIngestConfig
   */
  private mapToIngestConfig(data: any) {
    return {
      id: data.id || generateId('ingest-config'),
      name: data.name || 'RAGBits Ingest',
      description: data.description || 'Ingest documents into RAGBits',
      sourceType: data.sourceType || 'file',
      sourcePath: data.sourcePath || '',
      metadata: data.metadata || {},
      chunkSize: data.chunkSize || 1000,
      chunkOverlap: data.chunkOverlap || 200,
    };
  }

  /**
   * Maps node data to RAGBitsSearchConfig
   */
  private mapToSearchConfig(data: any) {
    return {
      id: data.id || generateId('search-config'),
      name: data.name || 'RAGBits Search',
      description: data.description || 'Perform semantic search with RAGBits',
      topK: data.topK || 5,
      scoreThreshold: data.scoreThreshold || 0.0,
      enableHybridSearch: data.enableHybridSearch || false,
      defaultFilters: data.defaultFilters || {},
    };
  }

  /**
   * Maps node data to RAGBitsGenerationConfig
   */
  private mapToGenerationConfig(data: any) {
    return {
      id: data.id || generateId('generation-config'),
      name: data.name || 'RAGBits Generation',
      description: data.description || 'Generate responses with RAGBits',
      llmModel: data.llmModel || 'gpt-4o',
      temperature: data.temperature || 0.7,
      maxTokens: data.maxTokens || 1000,
      systemPrompt: data.systemPrompt || '',
    };
  }

  /**
   * Maps node data to RAGBitsIndexConfig
   */
  private mapToIndexConfig(data: any) {
    return {
      id: data.id || generateId('index-config'),
      name: data.name || 'RAGBits Index',
      description: data.description || 'Manage RAGBits vector index',
      vectorStoreType: data.vectorStoreType || 'memory',
      embeddingModel: data.embeddingModel || 'text-embedding-3-small',
      autoRefresh: data.autoRefresh || false,
      refreshInterval: data.refreshInterval || 300,
    };
  }

  /**
   * Executes the entire workflow
   */
  async executeWorkflow(initialInput?: any): Promise<WorkflowExecutionResult[]> {
    this.logger.info(`Starting workflow execution: ${this.workflowConfig.name} (Execution ID: ${this.executionId})`);
    this.executionHistory = []; // Reset history

    // Topologically sort nodes to determine execution order
    const executionOrder = this.topologicalSort();
    this.logger.debug(`Execution order determined: ${executionOrder.join(', ')}`);

    // Execute nodes in order
    const nodeOutputs = new Map<string, any>();

    // Set initial input if provided
    if (initialInput) {
      // If there's a single starting node, pass the initial input to it
      const startingNodes = this.findStartingNodes(executionOrder);
      if (startingNodes.length > 0) {
        nodeOutputs.set(startingNodes[0], initialInput);
        this.logger.debug(`Set initial input for starting node: ${startingNodes[0]}`);
      }
    }

    for (const nodeId of executionOrder) {
      try {
        // Prepare input for the node based on connected outputs
        const input = this.prepareNodeInput(nodeId, nodeOutputs);

        // Execute the node
        const result = await this.executeNode(nodeId, input);

        // Store the output
        nodeOutputs.set(nodeId, result.output);

        // Add to execution history
        this.executionHistory.push(result);

        if (!result.success) {
          this.logger.error(`Node ${nodeId} failed: ${result.error}`);
          // Depending on requirements, we could stop execution or continue
          // For now, we'll continue to allow partial results
        } else {
          this.logger.debug(`Node ${nodeId} executed successfully in ${result.executionTime}ms`);
        }
      } catch (error) {
        const errorResult: WorkflowExecutionResult = {
          success: false,
          nodeId,
          output: null,
          executionTime: 0,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
        this.executionHistory.push(errorResult);
        this.logger.error(`Error executing node ${nodeId}: ${error}`);
      }
    }

    this.logger.info(`Workflow execution completed with ${this.executionHistory.length} nodes executed`);
    return this.executionHistory;
  }

  /**
   * Executes a single node
   */
  private async executeNode(nodeId: string, input: any): Promise<WorkflowExecutionResult> {
    const nodeInstance = this.nodeInstances.get(nodeId);
    if (!nodeInstance) {
      throw new Error(`Node instance not found: ${nodeId}`);
    }

    const startTime = Date.now();
    const operationId = generateId('node-operation');

    this.logger.debug(`Starting node execution ${operationId} for node: ${nodeId}`);

    try {
      // Execute with timeout
      const timeoutPromise = new Promise<WorkflowExecutionResult>((_, reject) => {
        setTimeout(() => reject(new Error(`Node ${nodeId} execution timed out`)), this.options.timeout);
      });

      const executionPromise = nodeInstance.action(input);

      const result = await Promise.race([executionPromise, timeoutPromise]);

      const executionTime = Date.now() - startTime;

      this.logger.debug(`Node ${nodeId} execution ${operationId} completed in ${executionTime}ms`);

      return {
        success: true,
        nodeId,
        output: result,
        executionTime,
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      this.logger.error(`Node ${nodeId} execution ${operationId} failed after ${executionTime}ms: ${errorMessage}`);

      return {
        success: false,
        nodeId,
        output: null,
        executionTime,
        error: errorMessage
      };
    }
  }

  /**
   * Prepares input for a node based on connected outputs
   */
  private prepareNodeInput(nodeId: string, nodeOutputs: Map<string, any>): any {
    // Find all edges that target this node
    const incomingEdges = this.workflowConfig.edges.filter(edge => edge.target === nodeId);

    if (incomingEdges.length === 0) {
      // No incoming edges, return empty input or initial input
      return {};
    }

    // If there's only one incoming edge, pass the output directly
    if (incomingEdges.length === 1) {
      const sourceNodeId = incomingEdges[0].source;
      const sourceOutput = nodeOutputs.get(sourceNodeId);
      return sourceOutput || {};
    }

    // If there are multiple incoming edges, combine the outputs
    const combinedInput: any = {};
    for (const edge of incomingEdges) {
      const sourceNodeId = edge.source;
      const sourceOutput = nodeOutputs.get(sourceNodeId);

      // Use the edge ID or a generated key to avoid conflicts
      const key = edge.sourceHandle || `input_${edge.source}`;
      combinedInput[key] = sourceOutput;
    }

    return combinedInput;
  }

  /**
   * Performs topological sort to determine execution order
   */
  private topologicalSort(): string[] {
    const visited = new Set<string>();
    const recursionStack = new Set<string>();
    const order: string[] = [];

    // Build adjacency list
    const adjList = new Map<string, string[]>();
    for (const node of this.workflowConfig.nodes) {
      adjList.set(node.id, []);
    }

    for (const edge of this.workflowConfig.edges) {
      const neighbors = adjList.get(edge.source) || [];
      neighbors.push(edge.target);
      adjList.set(edge.source, neighbors);
    }

    // Perform DFS for topological sort
    for (const node of this.workflowConfig.nodes) {
      if (!visited.has(node.id)) {
        this.topologicalSortUtil(node.id, visited, recursionStack, order, adjList);
      }
    }

    return order.reverse();
  }

  /**
   * Utility function for topological sort
   */
  private topologicalSortUtil(
    nodeId: string,
    visited: Set<string>,
    recursionStack: Set<string>,
    order: string[],
    adjList: Map<string, string[]>
  ): boolean {
    if (recursionStack.has(nodeId)) {
      throw new Error(`Cycle detected in workflow: ${nodeId}`);
    }

    if (visited.has(nodeId)) {
      return true;
    }

    visited.add(nodeId);
    recursionStack.add(nodeId);

    const neighbors = adjList.get(nodeId) || [];
    for (const neighbor of neighbors) {
      if (!this.topologicalSortUtil(neighbor, visited, recursionStack, order, adjList)) {
        return false;
      }
    }

    recursionStack.delete(nodeId);
    order.push(nodeId);
    return true;
  }

  /**
   * Finds starting nodes (nodes with no incoming edges)
   */
  private findStartingNodes(executionOrder: string[]): string[] {
    const nodeIds = new Set(this.workflowConfig.nodes.map(n => n.id));
    const targetNodes = new Set(this.workflowConfig.edges.map(e => e.target));

    const startingNodes: string[] = [];
    for (const nodeId of executionOrder) {
      if (!targetNodes.has(nodeId) && nodeIds.has(nodeId)) {
        startingNodes.push(nodeId);
      }
    }

    return startingNodes;
  }

  /**
   * Gets the execution history
   */
  getExecutionHistory(): WorkflowExecutionResult[] {
    return [...this.executionHistory];
  }

  /**
   * Gets the result of a specific node
   */
  getNodeResult(nodeId: string): WorkflowExecutionResult | undefined {
    return this.executionHistory.find(result => result.nodeId === nodeId);
  }

  /**
   * Logs messages based on the configured log level
   */
  private log(level: 'info' | 'debug' | 'warn' | 'error', message: string): void {
    if (!this.options.enableLogging) return;

    // Only log if the level is appropriate
    const levels = { error: 0, warn: 1, info: 2, debug: 3 };
    const currentLogLevel = levels[this.options.logLevel || 'info'];
    const messageLevel = levels[level];

    if (messageLevel <= currentLogLevel) {
      this.logger[level](message);
    }
  }

  /**
   * Resets the execution history
   */
  reset(): void {
    this.executionHistory = [];
    this.executionId = generateId('workflow-execution');
    this.logger = new Logger({ level: 'info', prefix: `RAGBitsWorkflowEngine-${this.executionId}` });
  }

  /**
   * Disposes of resources
   */
  async dispose(): Promise<void> {
    this.logger.info('Disposing workflow engine resources');

    // Dispose of node instances if they have a dispose method
    for (const [nodeId, instance] of this.nodeInstances) {
      if (typeof (instance as any).dispose === 'function') {
        try {
          await (instance as any).dispose();
          this.logger.debug(`Disposed node instance: ${nodeId}`);
        } catch (error) {
          this.log('warn', `Error disposing node ${nodeId}: ${error}`);
        }
      }
    }
    this.nodeInstances.clear();
    this.executionHistory = [];
    this.logger.info('Workflow engine disposed successfully');
  }

  /**
   * Add node instance management
   */
  getNodeInstance(nodeId: string): BaseBubble<any, any, any> | undefined {
    return this.nodeInstances.get(nodeId);
  }

  /**
   * Add node type detection
   */
  getNodeType(nodeId: string): string | undefined {
    const node = this.workflowConfig.nodes.find(n => n.id === nodeId);
    return node?.type;
  }

  /**
   * Add node configuration mapping
   */
  getNodeConfig(nodeId: string): any | undefined {
    const node = this.workflowConfig.nodes.find(n => n.id === nodeId);
    return node?.data;
  }

  /**
   * Add node initialization sequence
   */
  async initializeNode(nodeId: string): Promise<void> {
    const node = this.workflowConfig.nodes.find(n => n.id === nodeId);
    if (!node) {
      throw new Error(`Node not found: ${nodeId}`);
    }

    const nodeInstance = await this.createNodeInstance(node);
    this.nodeInstances.set(nodeId, nodeInstance);
    this.logger.debug(`Initialized node: ${nodeId}`);
  }

  /**
   * Add node validation
   */
  validateNode(nodeId: string): boolean {
    const nodeInstance = this.nodeInstances.get(nodeId);
    return !!nodeInstance && nodeInstance.isReady?.() !== false;
  }

  /**
   * Add node disposal handling
   */
  async disposeNode(nodeId: string): Promise<void> {
    const nodeInstance = this.nodeInstances.get(nodeId);
    if (nodeInstance && typeof (nodeInstance as any).dispose === 'function') {
      await (nodeInstance as any).dispose();
    }
    this.nodeInstances.delete(nodeId);
  }

  /**
   * Add node lifecycle management
   */
  async restartNode(nodeId: string): Promise<void> {
    await this.disposeNode(nodeId);
    await this.initializeNode(nodeId);
  }

  /**
   * Add workflow execution logic
   */
  async executePartialWorkflow(nodeIds: string[], initialInput?: any): Promise<WorkflowExecutionResult[]> {
    this.logger.info(`Executing partial workflow for nodes: ${nodeIds.join(', ')}`);

    // Create a temporary execution history for this partial execution
    const originalHistory = [...this.executionHistory];
    const partialResults: WorkflowExecutionResult[] = [];

    // Execute only the specified nodes
    for (const nodeId of nodeIds) {
      if (!this.nodeInstances.has(nodeId)) {
        const errorResult: WorkflowExecutionResult = {
          success: false,
          nodeId,
          output: null,
          executionTime: 0,
          error: `Node ${nodeId} not found in workflow`
        };
        partialResults.push(errorResult);
        continue;
      }

      try {
        // Prepare input for the node based on stored outputs
        const input = this.prepareNodeInput(nodeId, new Map(this.executionHistory.map(r => [r.nodeId, r.output])));

        const result = await this.executeNode(nodeId, input);
        partialResults.push(result);

        // Update the main execution history with the new result
        const existingIndex = this.executionHistory.findIndex(r => r.nodeId === nodeId);
        if (existingIndex !== -1) {
          this.executionHistory[existingIndex] = result;
        } else {
          this.executionHistory.push(result);
        }
      } catch (error) {
        const errorResult: WorkflowExecutionResult = {
          success: false,
          nodeId,
          output: null,
          executionTime: 0,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
        partialResults.push(errorResult);
      }
    }

    this.logger.info(`Partial workflow execution completed for ${partialResults.length} nodes`);
    return partialResults;
  }

  /**
   * Add execution order determination
   */
  determineExecutionOrder(): string[] {
    return this.topologicalSort();
  }

  /**
   * Add node execution sequence
   */
  async executeNodesInSequence(nodeIds: string[], initialInput?: any): Promise<WorkflowExecutionResult[]> {
    const results: WorkflowExecutionResult[] = [];
    let currentInput = initialInput;

    for (const nodeId of nodeIds) {
      try {
        const result = await this.executeNode(nodeId, currentInput);
        results.push(result);

        // Use the output of the current node as input for the next node
        if (result.success) {
          currentInput = result.output;
        }
      } catch (error) {
        const errorResult: WorkflowExecutionResult = {
          success: false,
          nodeId,
          output: null,
          executionTime: 0,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
        results.push(errorResult);
      }
    }

    return results;
  }

  /**
   * Add input preparation logic
   */
  prepareInputsForNode(nodeId: string): any {
    return this.prepareNodeInput(nodeId, new Map(this.executionHistory.map(r => [r.nodeId, r.output])));
  }

  /**
   * Add output collection logic
   */
  collectOutputs(): Map<string, any> {
    return new Map(this.executionHistory.map(r => [r.nodeId, r.output]));
  }

  /**
   * Add execution result tracking
   */
  trackExecutionResult(result: WorkflowExecutionResult): void {
    this.executionHistory.push(result);
  }

  /**
   * Add error handling during execution
   */
  async handleExecutionError(nodeId: string, error: any): Promise<WorkflowExecutionResult> {
    const errorResult: WorkflowExecutionResult = {
      success: false,
      nodeId,
      output: null,
      executionTime: 0,
      error: error instanceof Error ? error.message : 'Unknown error'
    };

    this.executionHistory.push(errorResult);
    return errorResult;
  }

  /**
   * Add partial execution continuation
   */
  async continueExecution(fromNode: string, input?: any): Promise<WorkflowExecutionResult[]> {
    const executionOrder = this.topologicalSort();
    const startIndex = executionOrder.indexOf(fromNode);

    if (startIndex === -1) {
      throw new Error(`Node ${fromNode} not found in execution order`);
    }

    const remainingNodes = executionOrder.slice(startIndex);
    return this.executeNodesInSequence(remainingNodes, input);
  }

  /**
   * Add topological sort implementation
   */
  getTopologicalSort(): string[] {
    return this.topologicalSort();
  }

  /**
   * Add cycle detection
   */
  hasCycles(): boolean {
    try {
      this.topologicalSort();
      return false;
    } catch (error) {
      return true;
    }
  }

  /**
   * Add execution order generation
   */
  generateExecutionOrder(): string[] {
    return this.topologicalSort();
  }

  /**
   * Add timeout handling
   */
  getTimeout(): number {
    return this.options.timeout;
  }

  /**
   * Add promise race for timeout
   */
  async executeWithTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(`Operation timed out after ${timeoutMs}ms`)), timeoutMs);
    });

    return Promise.race([promise, timeoutPromise]);
  }

  /**
   * Add execution time measurement
   */
  measureExecutionTime<T>(fn: () => Promise<T>): Promise<{ result: T; executionTime: number }> {
    const startTime = Date.now();
    return fn().then(result => {
      const executionTime = Date.now() - startTime;
      return { result, executionTime };
    });
  }

  /**
   * Add result formatting
   */
  formatResult(result: WorkflowExecutionResult): string {
    return `Node ${result.nodeId}: ${result.success ? 'SUCCESS' : 'FAILED'} (${result.executionTime}ms)`;
  }

  /**
   * Add logging for execution
   */
  logExecutionStatus(status: 'start' | 'complete' | 'error', details?: string): void {
    this.logger.info(`Workflow execution ${status}${details ? `: ${details}` : ''}`);
  }

  /**
   * Add metrics collection
   */
  collectExecutionMetrics(): {
    totalNodes: number;
    successfulNodes: number;
    failedNodes: number;
    totalExecutionTime: number;
    averageExecutionTime: number;
  } {
    const totalNodes = this.executionHistory.length;
    const successfulNodes = this.executionHistory.filter(r => r.success).length;
    const failedNodes = totalNodes - successfulNodes;
    const totalExecutionTime = this.executionHistory.reduce((sum, r) => sum + r.executionTime, 0);
    const averageExecutionTime = totalNodes > 0 ? totalExecutionTime / totalNodes : 0;

    return {
      totalNodes,
      successfulNodes,
      failedNodes,
      totalExecutionTime,
      averageExecutionTime
    };
  }

  /**
   * Add starting node detection
   */
  getStartingNodes(): string[] {
    return this.findStartingNodes(this.topologicalSort());
  }

  /**
   * Add execution history retrieval
   */
  getFilteredHistory(filter: (result: WorkflowExecutionResult) => boolean): WorkflowExecutionResult[] {
    return this.executionHistory.filter(filter);
  }

  /**
   * Add result filtering
   */
  getSuccessfulResults(): WorkflowExecutionResult[] {
    return this.getFilteredHistory(r => r.success);
  }

  /**
   * Add reset functionality
   */
  resetHistory(): void {
    this.executionHistory = [];
  }

  /**
   * Add result lookup by node ID
   */
  getResultByNodeId(nodeId: string): WorkflowExecutionResult | undefined {
    return this.executionHistory.find(r => r.nodeId === nodeId);
  }

  /**
   * Add execution statistics
   */
  getExecutionStats(): {
    totalExecutions: number;
    successRate: number;
    averageTimePerNode: number;
    lastExecutionTime: number;
  } {
    const total = this.executionHistory.length;
    const successful = this.executionHistory.filter(r => r.success).length;
    const totalTime = this.executionHistory.reduce((sum, r) => sum + r.executionTime, 0);
    const lastExecution = this.executionHistory.length > 0 ?
      Math.max(...this.executionHistory.map(r => r.executionTime)) : 0;

    return {
      totalExecutions: total,
      successRate: total > 0 ? (successful / total) * 100 : 0,
      averageTimePerNode: total > 0 ? totalTime / total : 0,
      lastExecutionTime: lastExecution
    };
  }

  /**
   * Add history validation
   */
  validateHistory(): boolean {
    return this.executionHistory.every(result =>
      typeof result.success === 'boolean' &&
      typeof result.nodeId === 'string' &&
      result.executionTime >= 0
    );
  }
}
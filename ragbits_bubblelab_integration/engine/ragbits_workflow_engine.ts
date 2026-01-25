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

export class RAGBitsWorkflowEngine {
  private workflowConfig: BubbleLabWorkflowConfig;
  private nodeInstances: Map<string, BaseBubble<any, any, any>>;
  private executionHistory: WorkflowExecutionResult[];
  private options: WorkflowExecutionOptions;

  constructor(workflowConfig: BubbleLabWorkflowConfig, options: WorkflowExecutionOptions = {}) {
    this.workflowConfig = workflowConfig;
    this.nodeInstances = new Map();
    this.executionHistory = [];
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
    this.log('info', `Initializing workflow: ${this.workflowConfig.name}`);

    // Validate the workflow configuration
    const validation = ConfigMapper.validateBubbleLabConfig(this.workflowConfig);
    if (!validation.isValid) {
      throw new Error(`Invalid workflow configuration: ${validation.errors.join(', ')}`);
    }

    // Create instances for each node
    for (const node of this.workflowConfig.nodes) {
      const nodeInstance = await this.createNodeInstance(node);
      this.nodeInstances.set(node.id, nodeInstance);
    }

    this.log('info', `Workflow initialized with ${this.nodeInstances.size} nodes`);
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
      id: data.id || '',
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
      id: data.id || '',
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
      id: data.id || '',
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
      id: data.id || '',
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
    this.log('info', `Starting workflow execution: ${this.workflowConfig.name}`);
    this.executionHistory = []; // Reset history

    // Topologically sort nodes to determine execution order
    const executionOrder = this.topologicalSort();

    // Execute nodes in order
    const nodeOutputs = new Map<string, any>();
    
    // Set initial input if provided
    if (initialInput) {
      // If there's a single starting node, pass the initial input to it
      const startingNodes = this.findStartingNodes(executionOrder);
      if (startingNodes.length > 0) {
        nodeOutputs.set(startingNodes[0], initialInput);
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
          this.log('error', `Node ${nodeId} failed: ${result.error}`);
          // Depending on requirements, we could stop execution or continue
          // For now, we'll continue to allow partial results
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
        this.log('error', `Error executing node ${nodeId}: ${error}`);
      }
    }

    this.log('info', `Workflow execution completed with ${this.executionHistory.length} nodes executed`);
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
    
    try {
      // Execute with timeout
      const timeoutPromise = new Promise<WorkflowExecutionResult>((_, reject) => {
        setTimeout(() => reject(new Error(`Node ${nodeId} execution timed out`)), this.options.timeout);
      });

      const executionPromise = nodeInstance.action(input);
      
      const result = await Promise.race([executionPromise, timeoutPromise]);
      
      const executionTime = Date.now() - startTime;
      
      return {
        success: true,
        nodeId,
        output: result,
        executionTime,
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      return {
        success: false,
        nodeId,
        output: null,
        executionTime,
        error: error instanceof Error ? error.message : 'Unknown error'
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
      console[level](`[RAGBitsWorkflowEngine - ${level.toUpperCase()}] ${message}`);
    }
  }

  /**
   * Resets the execution history
   */
  reset(): void {
    this.executionHistory = [];
  }

  /**
   * Disposes of resources
   */
  async dispose(): Promise<void> {
    // Dispose of node instances if they have a dispose method
    for (const [nodeId, instance] of this.nodeInstances) {
      if (typeof (instance as any).dispose === 'function') {
        try {
          await (instance as any).dispose();
        } catch (error) {
          this.log('warn', `Error disposing node ${nodeId}: ${error}`);
        }
      }
    }
    this.nodeInstances.clear();
    this.executionHistory = [];
  }
}
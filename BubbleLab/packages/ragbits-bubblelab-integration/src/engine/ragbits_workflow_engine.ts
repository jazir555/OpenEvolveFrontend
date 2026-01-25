/**
 * RAGBits Workflow Engine
 * Executes RAG workflows defined in BubbleLab
 */

import {
  BubbleLabWorkflowConfig,
  BubbleLabNode,
  BubbleLabEdge,
  WorkflowExecutionResult,
  WorkflowExecutionOptions,
  RAGBitsIngestConfig,
  RAGBitsSearchConfig,
  RAGBitsGenerationConfig,
  RAGBitsIndexConfig
} from '../types';
import { RAGBitsIngestBubble } from '../bubbles/ingest/RAGBitsIngestBubble';
import { RAGBitsSearchBubble } from '../bubbles/search/RAGBitsSearchBubble';
import { RAGBitsGenerationBubble } from '../bubbles/generation/RAGBitsGenerationBubble';
import { RAGBitsIndexBubble } from '../bubbles/index/RAGBitsIndexBubble';
import { RAGBitsDocumentProcessor } from '../integration/RagbitsProcessorIntegration';
import { ConfigMapper } from '../config/config_mapper';

/**
 * RAGBitsWorkflowEngine - Executes RAG workflows
 * Manages node instances, execution order, and result tracking
 */
export class RAGBitsWorkflowEngine {
  private workflowConfig: BubbleLabWorkflowConfig;
  private nodeInstances: Map<string, any>;
  private executionHistory: Map<string, any>;
  private executionOptions: WorkflowExecutionOptions;
  private processor: RAGBitsDocumentProcessor | null;
  private configMapper: ConfigMapper;
  private logger: Console;
  
  /**
   * Constructor
   * @param workflowConfig - BubbleLab workflow configuration
   * @param options - Execution options
   * @param processor - RAGBits document processor
   */
  constructor(
    workflowConfig: BubbleLabWorkflowConfig,
    options: WorkflowExecutionOptions = {},
    processor?: RAGBitsDocumentProcessor
  ) {
    this.workflowConfig = workflowConfig;
    this.executionOptions = {
      timeout: 30000, // 30 seconds default timeout
      maxRetries: 3,
      debug: false,
      logLevel: 'info',
      ...options
    };
    this.processor = processor || null;
    this.nodeInstances = new Map();
    this.executionHistory = new Map();
    this.configMapper = new ConfigMapper();
    this.logger = console;
  }
  
  /**
   * Initialize the workflow engine
   * Creates and initializes all node instances
   */
  public async initialize(): Promise<void> {
    this.log('info', `Initializing workflow: ${this.workflowConfig.name}`);
    
    try {
      // Create node instances
      for (const node of this.workflowConfig.nodes) {
        await this.createNodeInstance(node);
      }
      
      this.log('info', `Workflow initialized with ${this.nodeInstances.size} nodes`);
    } catch (error) {
      this.log('error', `Workflow initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }
  
  /**
   * Create node instance
   * @param node - BubbleLab node configuration
   */
  private async createNodeInstance(node: BubbleLabNode): Promise<void> {
    try {
      this.log('debug', `Creating node instance: ${node.id} (${node.type})`);
      
      let nodeInstance;
      
      switch (node.type) {
        case 'ragbits-ingest':
          nodeInstance = new RAGBitsIngestBubble(node.config as RAGBitsIngestConfig);
          break;
        case 'ragbits-search':
          nodeInstance = new RAGBitsSearchBubble(node.config as RAGBitsSearchConfig);
          break;
        case 'ragbits-generation':
          nodeInstance = new RAGBitsGenerationBubble(node.config as RAGBitsGenerationConfig);
          break;
        case 'ragbits-index':
          nodeInstance = new RAGBitsIndexBubble(node.config as RAGBitsIndexConfig);
          break;
        default:
          throw new Error(`Unknown node type: ${node.type}`);
      }
      
      // Initialize the node
      await nodeInstance.initialize(node, this.processor);
      
      // Store the instance
      this.nodeInstances.set(node.id, nodeInstance);
      
      this.log('debug', `Node ${node.id} initialized successfully`);
    } catch (error) {
      this.log('error', `Failed to create node ${node.id}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }
  
  /**
   * Execute the workflow
   * @returns Promise with workflow execution result
   */
  public async executeWorkflow(): Promise<WorkflowExecutionResult> {
    if (this.nodeInstances.size === 0) {
      throw new Error('Workflow not initialized. Call initialize() first.');
    }
    
    const startTime = Date.now();
    const executionId = `exec-${Date.now()}`;
    
    this.log('info', `Starting workflow execution: ${executionId}`);
    
    try {
      // Determine execution order using topological sort
      const executionOrder = this.topologicalSort();
      
      this.log('debug', `Execution order: ${executionOrder.join(' -> ')}`);
      
      // Execute nodes in order
      const nodeResults: Record<string, any> = {};
      let successfulNodes = 0;
      let failedNodes = 0;
      
      for (const nodeId of executionOrder) {
        try {
          const result = await this.executeNode(nodeId, nodeResults);
          nodeResults[nodeId] = result;
          successfulNodes++;
        } catch (error) {
          this.log('error', `Node ${nodeId} execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
          nodeResults[nodeId] = {
            error: error instanceof Error ? error.message : 'Unknown error'
          };
          failedNodes++;
          
          if (this.executionOptions.maxRetries && failedNodes > this.executionOptions.maxRetries) {
            throw new Error(`Maximum retries (${this.executionOptions.maxRetries}) exceeded`);
          }
        }
      }
      
      const endTime = Date.now();
      const totalDuration = endTime - startTime;
      
      const executionResult: WorkflowExecutionResult = {
        executionId,
        workflowId: this.workflowConfig.id,
        status: failedNodes === 0 ? 'success' : failedNodes === executionOrder.length ? 'failed' : 'partial',
        nodeResults,
        stats: {
          startTime: new Date(startTime),
          endTime: new Date(endTime),
          totalDuration,
          successfulNodes,
          failedNodes
        }
      };
      
      // Store execution history
      this.executionHistory.set(executionId, executionResult);
      
      this.log('info', `Workflow execution completed: ${executionResult.status}`);
      this.log('info', `Execution stats: ${successfulNodes} successful, ${failedNodes} failed, ${totalDuration}ms total`);
      
      return executionResult;
    } catch (error) {
      const endTime = Date.now();
      
      this.log('error', `Workflow execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      
      throw error;
    }
  }
  
  /**
   * Execute individual node
   * @param nodeId - Node ID to execute
   * @param previousResults - Results from previous nodes
   * @returns Promise with node execution result
   */
  private async executeNode(nodeId: string, previousResults: Record<string, any>): Promise<any> {
    const nodeInstance = this.nodeInstances.get(nodeId);
    
    if (!nodeInstance) {
      throw new Error(`Node ${nodeId} not found`);
    }
    
    this.log('debug', `Executing node: ${nodeId}`);
    
    // Prepare input for the node
    const nodeInput = await this.prepareNodeInput(nodeId, previousResults);
    
    // Execute with timeout
    return Promise.race([
      nodeInstance.action(nodeInput),
      new Promise((_, reject) => 
        setTimeout(() => 
          reject(new Error(`Node ${nodeId} execution timed out after ${this.executionOptions.timeout}ms`)),
          this.executionOptions.timeout
        )
      )
    ]);
  }
  
  /**
   * Prepare input for node execution
   * @param nodeId - Node ID
   * @param previousResults - Results from previous nodes
   * @returns Prepared input for the node
   */
  private async prepareNodeInput(nodeId: string, previousResults: Record<string, any>): Promise<any> {
    // Find incoming edges to this node
    const incomingEdges = this.workflowConfig.edges.filter(edge => edge.target === nodeId);
    
    if (incomingEdges.length === 0) {
      // No incoming edges - use default input
      return {};
    }
    
    // Collect outputs from previous nodes
    const nodeInput: Record<string, any> = {};
    
    for (const edge of incomingEdges) {
      const sourceNodeId = edge.source;
      const sourceResult = previousResults[sourceNodeId];
      
      if (sourceResult) {
        // Use the handle if specified, otherwise use the entire result
        if (edge.sourceHandle) {
          nodeInput[edge.sourceHandle] = sourceResult;
        } else {
          Object.assign(nodeInput, sourceResult);
        }
      }
    }
    
    this.log('debug', `Prepared input for node ${nodeId}: ${JSON.stringify(nodeInput)}`);
    
    return nodeInput;
  }
  
  /**
   * Topological sort for determining execution order
   * @returns Array of node IDs in execution order
   */
  private topologicalSort(): string[] {
    const nodes = this.workflowConfig.nodes;
    const edges = this.workflowConfig.edges;
    
    // Build adjacency list
    const adjacencyList: Record<string, string[]> = {};
    const inDegree: Record<string, number> = {};
    
    // Initialize
    nodes.forEach(node => {
      adjacencyList[node.id] = [];
      inDegree[node.id] = 0;
    });
    
    // Populate adjacency list and in-degree count
    edges.forEach(edge => {
      adjacencyList[edge.source].push(edge.target);
      inDegree[edge.target]++;
    });
    
    // Find starting nodes (nodes with no incoming edges)
    const queue: string[] = nodes
      .filter(node => inDegree[node.id] === 0)
      .map(node => node.id);
    
    const result: string[] = [];
    
    // Process nodes
    while (queue.length > 0) {
      const currentNode = queue.shift()!;
      result.push(currentNode);
      
      // Decrement in-degree for neighbors
      adjacencyList[currentNode].forEach(neighbor => {
        inDegree[neighbor]--;
        
        if (inDegree[neighbor] === 0) {
          queue.push(neighbor);
        }
      });
    }
    
    // Check for cycles
    if (result.length !== nodes.length) {
      throw new Error('Circular dependency detected in workflow - cannot determine execution order');
    }
    
    return result;
  }
  
  /**
   * Find starting nodes (nodes with no incoming edges)
   * @returns Array of starting node IDs
   */
  private findStartingNodes(): string[] {
    const allNodeIds = new Set(this.workflowConfig.nodes.map(node => node.id));
    const targetNodeIds = new Set(this.workflowConfig.edges.map(edge => edge.target));
    
    return Array.from(allNodeIds).filter(nodeId => !targetNodeIds.has(nodeId));
  }
  
  /**
   * Get execution history
   * @returns Array of execution results
   */
  public getExecutionHistory(): WorkflowExecutionResult[] {
    return Array.from(this.executionHistory.values());
  }
  
  /**
   * Get execution result by ID
   * @param executionId - Execution ID
   * @returns Execution result or null
   */
  public getExecutionResult(executionId: string): WorkflowExecutionResult | null {
    return this.executionHistory.get(executionId) || null;
  }
  
  /**
   * Reset execution history
   */
  public resetExecutionHistory(): void {
    this.executionHistory.clear();
  }
  
  /**
   * Dispose the workflow engine
   * Cleans up all node instances
   */
  public async dispose(): Promise<void> {
    this.log('info', 'Disposing workflow engine');
    
    // Dispose all node instances
    const disposePromises = Array.from(this.nodeInstances.values()).map(node => {
      return node.dispose().catch(error => {
        this.log('error', `Failed to dispose node: ${error instanceof Error ? error.message : 'Unknown error'}`);
      });
    });
    
    await Promise.all(disposePromises);
    
    this.nodeInstances.clear();
    this.executionHistory.clear();
    
    this.log('info', 'Workflow engine disposed');
  }
  
  /**
   * Log a message
   * @param level - Log level
   * @param message - Message to log
   * @param data - Additional data
   */
  private log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: any): void {
    if (this.executionOptions.logLevel === 'debug' || level !== 'debug') {
      const timestamp = new Date().toISOString();
      const logMessage = `[${timestamp}] [WorkflowEngine] [${level.toUpperCase()}] ${message}`;
      
      switch (level) {
        case 'debug':
          this.logger.debug(logMessage, data);
          break;
        case 'info':
          this.logger.info(logMessage, data);
          break;
        case 'warn':
          this.logger.warn(logMessage, data);
          break;
        case 'error':
          this.logger.error(logMessage, data);
          break;
      }
    }
  }
  
  /**
   * Get workflow configuration
   * @returns Workflow configuration
   */
  public getWorkflowConfig(): BubbleLabWorkflowConfig {
    return this.workflowConfig;
  }
  
  /**
   * Get execution options
   * @returns Execution options
   */
  public getExecutionOptions(): WorkflowExecutionOptions {
    return this.executionOptions;
  }
  
  /**
   * Set execution options
   * @param options - New execution options
   */
  public setExecutionOptions(options: Partial<WorkflowExecutionOptions>): void {
    this.executionOptions = { ...this.executionOptions, ...options };
  }
  
  /**
   * Get processor instance
   * @returns Current processor or null
   */
  public getProcessor(): RAGBitsDocumentProcessor | null {
    return this.processor;
  }
  
  /**
   * Set processor instance
   * @param processor - RAGBits document processor
   */
  public setProcessor(processor: RAGBitsDocumentProcessor): void {
    this.processor = processor;
    
    // Update all node instances with the new processor
    this.nodeInstances.forEach(node => {
      if (typeof node.setProcessor === 'function') {
        node.setProcessor(processor);
      }
    });
    
    this.log('info', 'Document processor updated for all nodes');
  }
}
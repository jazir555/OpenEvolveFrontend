/**
 * OpenEvolve gRPC Client for TypeScript
 * 
 * High-performance client for communicating with the OpenEvolve Python backend.
 * Provides streaming support, connection pooling, and automatic retries.
 */

import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { EventEmitter } from 'events';

import * as path from 'path';
import * as fs from 'fs';

// ============================================================================
// Type Definitions
// ============================================================================

export interface GRPCClientConfig {
  host: string;
  port: number;
  secure: boolean;
  credentials?: grpc.ChannelCredentials;
  
  // Connection pooling
  poolSize?: number;
  
  // Retry configuration
  maxRetries?: number;
  retryDelayMs?: number;
  retryableStatuses?: number[];
  
  // Timeouts
  defaultTimeoutMs?: number;
  connectTimeoutMs?: number;
  
  // Keepalive
  keepaliveTimeMs?: number;
  keepaliveTimeoutMs?: number;
  
  // Compression
  compression?: grpc.compressionAlgorithms;
  
  // Load balancing
  loadBalancingPolicy?: string;

  // Explicit path to the directory containing the .proto files.
  // Defaults to auto-detection relative to this module.
  protoDir?: string;
}

export interface ExecutionRequest {
  nodeType: string;
  inputs: Record<string, any>;
  config?: Record<string, any>;
  options?: ExecutionOptions;
}

export interface ExecutionOptions {
  timeoutMs?: number;
  enableStreaming?: boolean;
  enableCheckpointing?: boolean;
  maxRetries?: number;
  priority?: 'low' | 'normal' | 'high' | 'critical';
  labels?: Record<string, string>;
}

export interface ExecutionProgress {
  percent: number;
  message: string;
  stage?: string;
  timestamp: Date;
  metrics?: Record<string, any>;
}

export interface ExecutionResult {
  executionId: string;
  state: ExecutionState;
  result?: Record<string, any>;
  error?: ErrorDetails;
  progress?: ExecutionProgress;
  metrics?: Record<string, any>;
}

export type ExecutionState = 
  | 'PENDING'
  | 'RUNNING'
  | 'PAUSED'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED'
  | 'TIMEOUT';

export interface ErrorDetails {
  code: string;
  message: string;
  stackTrace?: string;
  context?: Record<string, any>;
  retryable: boolean;
  retryAfterMs?: number;
}

export interface NodeInfo {
  nodeId: string;
  nodeType: string;
  displayName: string;
  description: string;
  icon: string;
  category: string;
  version: string;
  tags: string[];
  capabilities: NodeCapabilities;
  parameterSchema?: Record<string, any>;
  inputSchema?: Record<string, any>;
  outputSchema?: Record<string, any>;
}

export interface NodeCapabilities {
  supportsStreaming: boolean;
  supportsCancellation: boolean;
  supportsProgress: boolean;
  supportsCheckpointing: boolean;
  supportsParallelExecution: boolean;
  maxTimeoutSeconds: number;
  requiredResources: string[];
}

export interface ServiceHealth {
  serviceName: string;
  status: 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY' | 'UNKNOWN';
  message: string;
  lastCheck: Date;
  responseTimeMs: number;
  metrics?: Record<string, any>;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: GRPCClientConfig = {
  host: 'localhost',
  port: 50051,
  secure: false,
  poolSize: 5,
  maxRetries: 3,
  retryDelayMs: 1000,
  retryableStatuses: [
    grpc.status.UNAVAILABLE,
    grpc.status.DEADLINE_EXCEEDED,
    grpc.status.RESOURCE_EXHAUSTED,
  ],
  defaultTimeoutMs: 60000,
  connectTimeoutMs: 10000,
  keepaliveTimeMs: 10000,
  keepaliveTimeoutMs: 5000,
  compression: grpc.compressionAlgorithms.gzip,
  loadBalancingPolicy: 'round_robin',
};

// ============================================================================
// gRPC Client Implementation
// ============================================================================

export class OpenEvolveGRPCClient extends EventEmitter {
  private config: GRPCClientConfig;
  private packageDefinition: any;
  private protoDescriptor: any;
  private nodeRegistry: any;
  private healthClient: any;
  
  // Connection management
  private channels: grpc.Client[] = [];
  private currentChannelIndex = 0;
  private isConnected = false;
  
  // Health check
  private healthCheckInterval?: NodeJS.Timeout;
  private lastHealthCheck?: ServiceHealth;
  
  // Active executions for cancellation
  private activeCalls = new Map<string, grpc.ClientReadableStream<any>>();

  constructor(config: Partial<GRPCClientConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.loadProto();
  }

  /**
   * Resolve the directory containing the .proto files.
   *
   * The proto/ directory lives one level above the typescript/ package root.
   * This must work for two different __dirname values:
   *   - running from source (typescript/)      -> ../proto
   *   - running from compiled output (dist/)   -> ../../proto
   * An explicit protoDir in the config always wins.
   */
  private resolveProtoDir(): string {
    const candidates = this.config.protoDir
      ? [this.config.protoDir]
      : [
          path.join(__dirname, '..', 'proto'),
          path.join(__dirname, '..', '..', 'proto'),
        ];

    for (const dir of candidates) {
      if (fs.existsSync(path.join(dir, 'common.proto'))) {
        return dir;
      }
    }

    throw new Error(
      `Unable to locate the OpenEvolve proto directory. Looked in: ${candidates.join(', ')}. ` +
        `Set 'protoDir' in the client config to the directory containing common.proto.`
    );
  }

  /**
   * Load protobuf definitions
   */
  private loadProto(): void {
    const protoDir = this.resolveProtoDir();
    
    const protoFiles = [
      path.join(protoDir, 'common.proto'),
      path.join(protoDir, 'nodes.proto'),
      path.join(protoDir, 'decomposition.proto'),
      path.join(protoDir, 'knowledge.proto'),
      path.join(protoDir, 'math.proto'),
      path.join(protoDir, 'gauntlet.proto'),
    ];

    this.packageDefinition = protoLoader.loadSync(protoFiles, {
      keepCase: true,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
      includeDirs: [protoDir],
    });

    this.protoDescriptor = grpc.loadPackageDefinition(this.packageDefinition);
    
    // Extract the service clients
    const openEvolveProto = (this.protoDescriptor as any).openevolve?.grpc;
    
    if (openEvolveProto?.NodeRegistry) {
      this.nodeRegistry = openEvolveProto.NodeRegistry;
    }
    
    // Load health check proto
    const healthProtoPath = path.join(protoDir, 'health.proto');
    const healthProto = grpc.loadPackageDefinition(
      protoLoader.loadSync(healthProtoPath, {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true,
        includeDirs: [protoDir],
      })
    );
    
    if ((healthProto as any).grpc?.health?.v1?.Health) {
      this.healthClient = (healthProto as any).grpc.health.v1.Health;
    }
  }

  /**
   * Connect to the gRPC server
   */
  public async connect(): Promise<void> {
    if (this.isConnected) {
      return;
    }

    const address = `${this.config.host}:${this.config.port}`;
    
    // Create channel credentials
    const credentials = this.config.secure
      ? (this.config.credentials || grpc.credentials.createSsl())
      : grpc.credentials.createInsecure();

    // Create channel options
    const options: grpc.ChannelOptions = {
      'grpc.max_send_message_length': 50 * 1024 * 1024, // 50MB
      'grpc.max_receive_message_length': 50 * 1024 * 1024, // 50MB
      'grpc.keepalive_time_ms': this.config.keepaliveTimeMs,
      'grpc.keepalive_timeout_ms': this.config.keepaliveTimeoutMs,
      'grpc.keepalive_permit_without_calls': 1,
      'grpc.http2.max_pings_without_data': 0,
      'grpc.http2.min_time_between_pings_ms': 10000,
      'grpc.http2.min_ping_interval_without_data_ms': 5000,
      'grpc.service_config': JSON.stringify({
        loadBalancingConfig: [{ [this.config.loadBalancingPolicy!]: {} }],
      }),
    };

    // Create connection pool
    for (let i = 0; i < (this.config.poolSize || 1); i++) {
      const channel = new this.nodeRegistry(address, credentials, options);
      this.channels.push(channel);
    }

    // Wait for connection
    await this.waitForReady();
    
    this.isConnected = true;
    this.emit('connected');
    
    // Start health checks
    this.startHealthChecks();
    
    // Setup reconnection handling
    this.setupReconnection();
  }

  /**
   * Wait for channel to be ready
   */
  private async waitForReady(): Promise<void> {
    const deadline = new Date(Date.now() + (this.config.connectTimeoutMs || 10000));
    
    for (const channel of this.channels) {
      await new Promise<void>((resolve, reject) => {
        channel.waitForReady(deadline, (err) => {
          if (err) {
            reject(err);
          } else {
            resolve();
          }
        });
      });
    }
  }

  /**
   * Get next channel from pool (round-robin)
   */
  private getChannel(): grpc.Client {
    if (this.channels.length === 0) {
      throw new Error(
        'No gRPC channel available: call connect() before issuing requests.'
      );
    }
    const channel = this.channels[this.currentChannelIndex];
    this.currentChannelIndex = (this.currentChannelIndex + 1) % this.channels.length;
    return channel;
  }

  /**
   * Start periodic health checks
   */
  private startHealthChecks(): void {
    if (!this.healthClient) return;

    this.healthCheckInterval = setInterval(async () => {
      try {
        const health = await this.checkHealth();
        this.lastHealthCheck = health;
        this.emit('health', health);
        
        if (health.status !== 'HEALTHY') {
          this.emit('degraded', health);
        }
      } catch (err) {
        this.emit('healthError', err);
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Setup automatic reconnection
   */
  private setupReconnection(): void {
    for (const channel of this.channels) {
      channel.getChannel().watchConnectivityState(
        channel.getChannel().getConnectivityState(true),
        Infinity,
        (err) => {
          if (err) {
            this.emit('disconnected', err);
            this.isConnected = false;
            this.reconnect();
          }
        }
      );
    }
  }

  /**
   * Reconnect to server
   */
  private async reconnect(): Promise<void> {
    this.emit('reconnecting');
    
    try {
      await this.close();
      await this.connect();
      this.emit('reconnected');
    } catch (err) {
      this.emit('reconnectFailed', err);
      // Retry after delay
      setTimeout(() => this.reconnect(), this.config.retryDelayMs);
    }
  }

  /**
   * Close all connections
   */
  public async close(): Promise<void> {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    // Cancel all active calls
    for (const [id, call] of this.activeCalls) {
      call.cancel();
      this.activeCalls.delete(id);
    }

    for (const channel of this.channels) {
      channel.close();
    }
    
    this.channels = [];
    this.isConnected = false;
    this.emit('disconnected');
  }

  // =========================================================================
  // Public API Methods
  // =========================================================================

  /**
   * List all available nodes
   */
  public async listNodes(category?: string): Promise<NodeInfo[]> {
    const request = {
      metadata: this.createRequestMetadata(),
      category: category || '',
    };

    const response = await this.callWithRetry('ListNodes', request);
    return response.nodes.map((n: any) => this.mapNodeInfo(n));
  }

  /**
   * Get detailed information about a specific node
   */
  public async getNodeSchema(nodeType: string): Promise<NodeInfo> {
    const request = {
      metadata: this.createRequestMetadata(),
      node_type: nodeType,
    };

    const response = await this.callWithRetry('GetNodeSchema', request);
    return this.mapNodeInfo(response.node_info);
  }

  /**
   * Execute a node synchronously
   */
  public async executeNode(request: ExecutionRequest): Promise<ExecutionResult> {
    const grpcRequest = {
      metadata: this.createRequestMetadata(),
      node_type: request.nodeType,
      inputs: this.toStruct(request.inputs),
      config: this.toStruct(request.config || {}),
      options: this.mapExecutionOptions(request.options),
    };

    const response = await this.callWithRetry(
      'ExecuteNode',
      grpcRequest,
      request.options?.timeoutMs || this.config.defaultTimeoutMs
    );

    return this.mapExecutionResult(response);
  }

  /**
   * Execute a node with streaming progress updates
   */
  public executeNodeStreaming(
    request: ExecutionRequest,
    onProgress: (progress: ExecutionProgress) => void
  ): Promise<ExecutionResult> {
    return new Promise((resolve, reject) => {
      const grpcRequest = {
        metadata: this.createRequestMetadata(),
        node_type: request.nodeType,
        inputs: this.toStruct(request.inputs),
        config: this.toStruct(request.config || {}),
        options: this.mapExecutionOptions(request.options),
      };

      const channel = this.getChannel() as any;
      const call = channel.ExecuteNodeStreaming(grpcRequest);
      
      const executionId = grpcRequest.metadata.request_id;
      this.activeCalls.set(executionId, call);

      let finalResult: ExecutionResult | null = null;

      call.on('data', (update: any) => {
        if (update.progress) {
          onProgress(this.mapProgress(update.progress));
        }
        
        if (update.state === 'EXECUTION_STATE_COMPLETED') {
          finalResult = {
            executionId: update.execution_id,
            state: 'COMPLETED',
            result: update.partial_result ? this.fromStruct(update.partial_result) : undefined,
            progress: update.progress ? this.mapProgress(update.progress) : undefined,
          };
        } else if (update.state === 'EXECUTION_STATE_FAILED') {
          finalResult = {
            executionId: update.execution_id,
            state: 'FAILED',
            error: update.error ? this.mapError(update.error) : undefined,
          };
        }
      });

      call.on('error', (err: grpc.ServiceError) => {
        this.activeCalls.delete(executionId);
        
        if (err.code === grpc.status.CANCELLED) {
          resolve({
            executionId,
            state: 'CANCELLED',
          });
        } else {
          reject(this.mapGRPCError(err));
        }
      });

      call.on('end', () => {
        this.activeCalls.delete(executionId);
        
        if (finalResult) {
          resolve(finalResult);
        } else {
          reject(new Error('Stream ended without result'));
        }
      });
    });
  }

  /**
   * Cancel a running execution
   */
  public async cancelExecution(executionId: string): Promise<boolean> {
    // Cancel active call if streaming
    const call = this.activeCalls.get(executionId);
    if (call) {
      call.cancel();
      this.activeCalls.delete(executionId);
      return true;
    }

    // Otherwise send cancel request
    const request = {
      metadata: this.createRequestMetadata(),
      execution_id: executionId,
      reason: 'User requested cancellation',
    };

    const response = await this.callWithRetry('CancelExecution', request);
    return response.success;
  }

  /**
   * Get execution status
   */
  public async getExecutionStatus(executionId: string): Promise<ExecutionResult> {
    const request = {
      metadata: this.createRequestMetadata(),
      execution_id: executionId,
    };

    const response = await this.callWithRetry('GetExecutionStatus', request);
    return this.mapExecutionResult(response);
  }

  /**
   * Check server health
   */
  public async checkHealth(): Promise<ServiceHealth> {
    if (!this.healthClient) {
      throw new Error('Health client not available');
    }

    const health = new this.healthClient(
      `${this.config.host}:${this.config.port}`,
      this.config.secure ? grpc.credentials.createSsl() : grpc.credentials.createInsecure()
    );

    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      
      health.Check({}, (err: grpc.ServiceError | null, response: any) => {
        if (err) {
          reject(err);
          return;
        }

        const responseTime = Date.now() - startTime;
        
        resolve({
          serviceName: 'openevolve.grpc.NodeRegistry',
          status: response.status === 'SERVING' ? 'HEALTHY' : 'UNHEALTHY',
          message: `Health status: ${response.status}`,
          lastCheck: new Date(),
          responseTimeMs: responseTime,
        });
      });
    });
  }

  /**
   * Get the result of the last health check
   */
  public getLastHealthCheck(): ServiceHealth | undefined {
    return this.lastHealthCheck;
  }

  // =========================================================================
  // Helper Methods
  // =========================================================================

  /**
   * Create request metadata
   */
  private createRequestMetadata(): any {
    return {
      request_id: this.generateRequestId(),
      timestamp: new Date().toISOString(),
      client_version: '2.0.0-grpc-ts',
    };
  }

  /**
   * Generate unique request ID
   */
  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Convert JS object to protobuf Struct
   */
  private toStruct(obj: Record<string, any>): any {
    // This would use the actual protobuf Struct conversion
    // For now, return as-is (grpc-js handles this)
    return obj;
  }

  /**
   * Convert protobuf Struct to JS object
   */
  private fromStruct(struct: any): Record<string, any> {
    // This would use the actual protobuf Struct conversion
    return struct;
  }

  /**
   * Map execution options
   */
  private mapExecutionOptions(options?: ExecutionOptions): any {
    if (!options) return {};
    
    return {
      timeout_seconds: options.timeoutMs ? Math.floor(options.timeoutMs / 1000) : undefined,
      enable_streaming: options.enableStreaming,
      enable_checkpointing: options.enableCheckpointing,
      max_retries: options.maxRetries,
      execution_priority: options.priority,
      labels: options.labels,
    };
  }

  /**
   * Map NodeInfo from gRPC response
   */
  private mapNodeInfo(info: any): NodeInfo {
    return {
      nodeId: info.node_id,
      nodeType: info.node_type,
      displayName: info.display_name,
      description: info.description,
      icon: info.icon,
      category: info.category,
      version: info.version,
      tags: info.tags || [],
      capabilities: {
        supportsStreaming: info.capabilities?.supports_streaming || false,
        supportsCancellation: info.capabilities?.supports_cancellation || false,
        supportsProgress: info.capabilities?.supports_progress || false,
        supportsCheckpointing: info.capabilities?.supports_checkpointing || false,
        supportsParallelExecution: info.capabilities?.supports_parallel_execution || false,
        maxTimeoutSeconds: info.capabilities?.max_timeout_seconds || 300,
        requiredResources: info.capabilities?.required_resources || [],
      },
      parameterSchema: info.parameter_schema,
      inputSchema: info.input_schema,
      outputSchema: info.output_schema,
    };
  }

  /**
   * Map ExecutionResult from gRPC response
   */
  private mapExecutionResult(response: any): ExecutionResult {
    const stateMap: Record<string, ExecutionState> = {
      'EXECUTION_STATE_PENDING': 'PENDING',
      'EXECUTION_STATE_RUNNING': 'RUNNING',
      'EXECUTION_STATE_PAUSED': 'PAUSED',
      'EXECUTION_STATE_COMPLETED': 'COMPLETED',
      'EXECUTION_STATE_FAILED': 'FAILED',
      'EXECUTION_STATE_CANCELLED': 'CANCELLED',
      'EXECUTION_STATE_TIMEOUT': 'TIMEOUT',
    };

    return {
      executionId: response.execution_id,
      state: stateMap[response.state] || 'FAILED',
      result: response.result ? this.fromStruct(response.result) : undefined,
      error: response.error ? this.mapError(response.error) : undefined,
      progress: response.final_progress ? this.mapProgress(response.final_progress) : undefined,
      metrics: response.execution_metrics ? this.fromStruct(response.execution_metrics) : undefined,
    };
  }

  /**
   * Map Progress from gRPC response
   */
  private mapProgress(progress: any): ExecutionProgress {
    return {
      percent: progress.percent,
      message: progress.message,
      stage: progress.stage,
      timestamp: new Date(progress.timestamp),
      metrics: progress.metrics ? this.fromStruct(progress.metrics) : undefined,
    };
  }

  /**
   * Map Error from gRPC response
   */
  private mapError(error: any): ErrorDetails {
    return {
      code: error.error_code,
      message: error.message,
      stackTrace: error.stack_trace,
      context: error.context ? this.fromStruct(error.context) : undefined,
      retryable: error.retryable,
      retryAfterMs: error.retry_after_seconds ? error.retry_after_seconds * 1000 : undefined,
    };
  }

  /**
   * Map gRPC error to application error
   */
  private mapGRPCError(err: grpc.ServiceError): Error {
    const message = err.details || err.message || 'Unknown error';
    const error = new Error(message) as Error & { code: number; metadata: any };
    error.code = err.code;
    error.metadata = err.metadata;
    return error;
  }

  /**
   * Call gRPC method with retry logic
   */
  private async callWithRetry(
    method: string,
    request: any,
    timeoutMs?: number
  ): Promise<any> {
    const maxRetries = this.config.maxRetries || 3;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await this.callMethod(method, request, timeoutMs);
      } catch (err: any) {
        lastError = err;
        
        // Check if error is retryable
        if (!this.isRetryableError(err)) {
          throw err;
        }

        // Wait before retry
        if (attempt < maxRetries - 1) {
          const delay = (this.config.retryDelayMs || 1000) * Math.pow(2, attempt);
          await this.sleep(delay);
        }
      }
    }

    throw lastError;
  }

  /**
   * Call gRPC method
   */
  private callMethod(
    method: string,
    request: any,
    timeoutMs?: number
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const channel = this.getChannel();
      const deadline = timeoutMs 
        ? new Date(Date.now() + timeoutMs) 
        : undefined;

      (channel as any)[method](request, { deadline }, (err: grpc.ServiceError | null, response: any) => {
        if (err) {
          reject(this.mapGRPCError(err));
        } else {
          resolve(response);
        }
      });
    });
  }

  /**
   * Check if error is retryable
   */
  private isRetryableError(err: any): boolean {
    const retryableStatuses = this.config.retryableStatuses || [];
    return retryableStatuses.includes(err.code);
  }

  /**
   * Sleep for given milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Create a gRPC client with default configuration
 */
export function createGRPCClient(config?: Partial<GRPCClientConfig>): OpenEvolveGRPCClient {
  return new OpenEvolveGRPCClient(config);
}

/**
 * Quick execute helper for simple use cases
 */
export async function quickExecute(
  nodeType: string,
  inputs: Record<string, any>,
  config?: Partial<GRPCClientConfig>
): Promise<ExecutionResult> {
  const client = createGRPCClient(config);
  
  try {
    await client.connect();
    return await client.executeNode({ nodeType, inputs });
  } finally {
    await client.close();
  }
}

// Export everything
export default OpenEvolveGRPCClient;

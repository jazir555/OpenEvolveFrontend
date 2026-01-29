/**
 * Anti-Corruption Layer (ACL) for OpenEvolve Integration
 *
 * Implements the Anti-Corruption Layer pattern to protect OpenEvolve
 * from external system changes. Handles request/response normalization,
 * protocol translation, data format conversion, and error mapping.
 *
 * Following the Federation Constitution's LAW OF THE "AIR GAP":
 * - No direct imports from core-projects
 * - Runtime truth verification
 * - Explicit configuration
 * - Idempotent operations
 * - UTC timestamps
 */

import { z } from 'zod';
import CanonicalModels, {
  CanonicalUser,
  CanonicalService,
  CanonicalWorkflow,
  CanonicalKnowledgeDocument,
  CanonicalLogEntry,
} from '../schemas/canonical-models';

// ============================================================================
// PROTOCOL ADAPTERS
// ============================================================================

/**
 * Protocol Adapter Interface
 */
interface IProtocolAdapter {
  normalizeRequest(request: unknown): Promise<CanonicalRequest>;
  normalizeResponse(response: unknown): Promise<CanonicalResponse>;
  transformErrors(error: unknown): CanonicalError;
}

/**
 * HTTP Protocol Adapter
 */
class HttpProtocolAdapter implements IProtocolAdapter {
  async normalizeRequest(request: unknown): Promise<CanonicalRequest> {
    const startTime = Date.now();

    try {
      if (this.isHttpRequest(request)) {
        return {
          protocol: 'http',
          method: request.method || 'GET',
          headers: this.normalizeHeaders(request.headers),
          body: request.body,
          query: request.query || {},
          path: request.path || '/',
          timestamp: new Date().toISOString(),
          metadata: {
            contentType: request.headers?.['content-type'] || 'application/json',
            contentLength: request.headers?.['content-length'],
            userAgent: request.headers?.['user-agent'],
          },
        };
      }

      throw new Error('Invalid HTTP request format');
    } catch (error) {
      throw new Error(`HTTP request normalization failed: ${error}`);
    }
  }

  async normalizeResponse(response: unknown): Promise<CanonicalResponse> {
    try {
      if (this.isHttpResponse(response)) {
        return {
          protocol: 'http',
          statusCode: response.status || response.statusCode,
          statusText: response.statusText || '',
          headers: this.normalizeHeaders(response.headers),
          body: response.data || response.body,
          timestamp: new Date().toISOString(),
          metadata: {
            contentType: response.headers?.['content-type'] || 'application/json',
            contentLength: response.headers?.['content-length'],
            responseTime: response.headers?.['x-response-time'],
          },
        };
      }

      throw new Error('Invalid HTTP response format');
    } catch (error) {
      throw new Error(`HTTP response normalization failed: ${error}`);
    }
  }

  transformErrors(error: unknown): CanonicalError {
    if (error instanceof Error) {
      return {
        code: 'HTTP_ERROR',
        message: error.message,
        type: 'external',
        details: {
          name: error.name,
          stack: error.stack,
        },
        timestamp: new Date().toISOString(),
        service: 'http',
      };
    }

    return {
      code: 'UNKNOWN_HTTP_ERROR',
      message: 'Unknown HTTP error occurred',
      type: 'external',
      timestamp: new Date().toISOString(),
      service: 'http',
    };
  }

  private isHttpRequest(data: any): boolean {
    return data && (data.method || data.url || data.headers);
  }

  private isHttpResponse(data: any): boolean {
    return data && (typeof data.status === 'number' || data.statusCode || data.data);
  }

  private normalizeHeaders(headers: Record<string, string | string[] | undefined>): Record<string, string> {
    const normalized: Record<string, string> = {};

    for (const [key, value] of Object.entries(headers)) {
      if (value !== undefined) {
        normalized[key.toLowerCase()] = Array.isArray(value) ? value[0] : value;
      }
    }

    return normalized;
  }
}

/**
 * gRPC Protocol Adapter
 */
class GrpcProtocolAdapter implements IProtocolAdapter {
  async normalizeRequest(request: unknown): Promise<CanonicalRequest> {
    try {
      if (this.isGrpcRequest(request)) {
        return {
          protocol: 'grpc',
          method: request.method || '',
          service: request.service || '',
          headers: this.normalizeMetadata(request.metadata),
          body: request.message,
          timestamp: new Date().toISOString(),
        };
      }

      throw new Error('Invalid gRPC request format');
    } catch (error) {
      throw new Error(`gRPC request normalization failed: ${error}`);
    }
  }

  async normalizeResponse(response: unknown): Promise<CanonicalResponse> {
    try {
      if (this.isGrpcResponse(response)) {
        return {
          protocol: 'grpc',
          statusCode: response.code || 0,
          statusText: response.details || '',
          headers: this.normalizeMetadata(response.metadata),
          body: response.message,
          timestamp: new Date().toISOString(),
        };
      }

      throw new Error('Invalid gRPC response format');
    } catch (error) {
      throw new Error(`gRPC response normalization failed: ${error}`);
    }
  }

  transformErrors(error: unknown): CanonicalError {
    return {
      code: 'GRPC_ERROR',
      message: error instanceof Error ? error.message : 'Unknown gRPC error',
      type: 'external',
      timestamp: new Date().toISOString(),
      service: 'grpc',
    };
  }

  private isGrpcRequest(data: any): boolean {
    return data && (data.service || data.method || data.message);
  }

  private isGrpcResponse(data: any): boolean {
    return data && (typeof data.code === 'number' || data.message);
  }

  private normalizeMetadata(metadata: Record<string, any>): Record<string, string> {
    const normalized: Record<string, string> = {};

    for (const [key, value] of Object.entries(metadata)) {
      normalized[key.toLowerCase()] = String(value);
    }

    return normalized;
  }
}

// ============================================================================
// DATA TRANSFORMERS
// ============================================================================

/**
 * Data Transformer Interface
 */
interface IDataTransformer {
  toCanonical(source: unknown, sourceType: string): Promise<CanonicalData>;
  fromCanonical(canonical: CanonicalData, targetType: string): Promise<unknown>;
}

/**
 * Knowledge Data Transformer
 */
class KnowledgeDataTransformer implements IDataTransformer {
  async toCanonical(source: unknown, sourceType: string): Promise<CanonicalData> {
    switch (sourceType) {
      case 'qdrant':
        return this.qdrantToCanonical(source);
      case 'elasticsearch':
        return this.elasticsearchToCanonical(source);
      case 'bedrock':
        return this.bedrockToCanonical(source);
      default:
        throw new Error(`Unknown source type: ${sourceType}`);
    }
  }

  async fromCanonical(canonical: CanonicalData, targetType: string): Promise<unknown> {
    switch (targetType) {
      case 'qdrant':
        return this.canonicalToQdrant(canonical);
      case 'elasticsearch':
        return this.canonicalToElasticsearch(canonical);
      default:
        throw new Error(`Unknown target type: ${targetType}`);
    }
  }

  private async qdrantToCanonical(source: unknown): Promise<CanonicalData> {
    if (!this.isQdrantPoint(source)) {
      throw new Error('Invalid Qdrant point format');
    }

    const validation = CanonicalModels.validateCanonical(
      CanonicalModels.CanonicalKnowledgeDocumentSchema,
      CanonicalModels.qdrantPointToCanonical(source)
    );

    if (!validation.success) {
      throw new Error(`Qdrant to canonical transformation failed: ${validation.error}`);
    }

    return validation.data;
  }

  private async elasticsearchToCanonical(source: unknown): Promise<CanonicalData> {
    if (!this.isElasticsearchDoc(source)) {
      throw new Error('Invalid Elasticsearch document format');
    }

    const validation = CanonicalModels.validateCanonical(
      CanonicalModels.CanonicalKnowledgeDocumentSchema,
      CanonicalModels.elasticsearchDocToCanonical(source)
    );

    if (!validation.success) {
      throw new Error(`Elasticsearch to canonical transformation failed: ${validation.error}`);
    }

    return validation.data;
  }

  private async bedrockToCanonical(source: unknown): Promise<CanonicalData> {
    // Bedrock-specific transformation
    const bedrockData = source as any;
    return {
      id: bedrockData.metadata?.uri || bedrockData.content?.text?.substring(0, 50),
      content: bedrockData.content?.text || '',
      metadata: {
        source: 'bedrock',
        type: 'document',
        tags: [],
        timestamp: new Date().toISOString(),
        language: 'en',
      },
    };
  }

  private async canonicalToQdrant(canonical: CanonicalData): Promise<unknown> {
    const doc = canonical as CanonicalKnowledgeDocument;
    return {
      id: doc.id,
      vector: doc.embedding,
      payload: {
        content: doc.content,
        source: doc.metadata.source,
        type: doc.metadata.type,
        tags: doc.metadata.tags,
        timestamp: doc.metadata.timestamp,
        language: doc.metadata.language,
      },
    };
  }

  private async canonicalToElasticsearch(canonical: CanonicalData): Promise<unknown> {
    const doc = canonical as CanonicalKnowledgeDocument;
    return {
      content: doc.content,
      source: doc.metadata.source,
      type: doc.metadata.type,
      tags: doc.metadata.tags,
      timestamp: doc.metadata.timestamp,
      language: doc.metadata.language,
    };
  }

  private isQdrantPoint(data: any): boolean {
    return data && (data.id || data.vector || data.payload);
  }

  private isElasticsearchDoc(data: any): boolean {
    return data && (data._id || data._source || data._index);
  }
}

// ============================================================================
// CIRCUIT BREAKER
// ============================================================================

/**
 * Circuit Breaker States
 */
enum CircuitBreakerState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

/**
 * Circuit Breaker Configuration
 */
interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;
  halfOpenAttempts: number;
}

/**
 * Circuit Breaker Implementation
 */
class CircuitBreaker {
  private state: CircuitBreakerState = CircuitBreakerState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private halfOpenAttemptCount = 0;

  constructor(private config: CircuitBreakerConfig) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === CircuitBreakerState.OPEN) {
      if (this.shouldAttemptReset()) {
        this.state = CircuitBreakerState.HALF_OPEN;
        this.halfOpenAttemptCount = 0;
      } else {
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;

    if (this.state === CircuitBreakerState.HALF_OPEN) {
      this.halfOpenAttemptCount++;
      if (this.halfOpenAttemptCount >= this.config.halfOpenAttempts) {
        this.state = CircuitBreakerState.CLOSED;
      }
    }
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.failureCount >= this.config.failureThreshold) {
      this.state = CircuitBreakerState.OPEN;
    }
  }

  private shouldAttemptReset(): boolean {
    return Date.now() - this.lastFailureTime > this.config.timeout;
  }

  getState(): CircuitBreakerState {
    return this.state;
  }
}

// ============================================================================
// ANTI-CORRUPTION LAYER
// ============================================================================

/**
 * ACL Configuration
 */
interface ACLConfig {
  protocolAdapters: Map<string, IProtocolAdapter>;
  dataTransformers: Map<string, IDataTransformer>;
  circuitBreakers: Map<string, CircuitBreaker>;
  enableValidation: boolean;
  enableMetrics: boolean;
}

/**
 * Anti-Corruption Layer Main Class
 */
export class AntiCorruptionLayer {
  private config: ACLConfig;

  constructor(config?: Partial<ACLConfig>) {
    const protocolAdapters = new Map<string, IProtocolAdapter>();
    protocolAdapters.set('http', new HttpProtocolAdapter());
    protocolAdapters.set('grpc', new GrpcProtocolAdapter());

    const dataTransformers = new Map<string, IDataTransformer>();
    dataTransformers.set('knowledge', new KnowledgeDataTransformer());

    const circuitBreakers = new Map<string, CircuitBreaker>();
    circuitBreakers.set('default', new CircuitBreaker({
      failureThreshold: 5,
      successThreshold: 2,
      timeout: 60000,
      halfOpenAttempts: 3,
    }));

    this.config = {
      protocolAdapters,
      dataTransformers,
      circuitBreakers,
      enableValidation: true,
      enableMetrics: true,
      ...config,
    };
  }

  /**
   * Normalize incoming request
   */
  async normalizeRequest(request: unknown, protocol: string): Promise<CanonicalRequest> {
    const adapter = this.config.protocolAdapters.get(protocol);
    if (!adapter) {
      throw new Error(`No protocol adapter found for: ${protocol}`);
    }

    return adapter.normalizeRequest(request);
  }

  /**
   * Normalize outgoing response
   */
  async normalizeResponse(response: unknown, protocol: string): Promise<CanonicalResponse> {
    const adapter = this.config.protocolAdapters.get(protocol);
    if (!adapter) {
      throw new Error(`No protocol adapter found for: ${protocol}`);
    }

    return adapter.normalizeResponse(response);
  }

  /**
   * Transform data to canonical format
   */
  async toCanonical(source: unknown, sourceType: string, transformer: string): Promise<CanonicalData> {
    const dataTransformer = this.config.dataTransformers.get(transformer);
    if (!dataTransformer) {
      throw new Error(`No data transformer found for: ${transformer}`);
    }

    return dataTransformer.toCanonical(source, sourceType);
  }

  /**
   * Transform data from canonical format
   */
  async fromCanonical(canonical: CanonicalData, targetType: string, transformer: string): Promise<unknown> {
    const dataTransformer = this.config.dataTransformers.get(transformer);
    if (!dataTransformer) {
      throw new Error(`No data transformer found for: ${transformer}`);
    }

    return dataTransformer.fromCanonical(canonical, targetType);
  }

  /**
   * Execute operation with circuit breaker protection
   */
  async executeWithCircuitBreaker<T>(
    operation: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const circuitBreaker = this.config.circuitBreakers.get(operation) ||
                          this.config.circuitBreakers.get('default');

    if (!circuitBreaker) {
      return fn();
    }

    return circuitBreaker.execute(fn);
  }

  /**
   * Transform errors
   */
  transformError(error: unknown, protocol: string): CanonicalError {
    const adapter = this.config.protocolAdapters.get(protocol);
    if (!adapter) {
      return {
        code: 'UNKNOWN_ERROR',
        message: error instanceof Error ? error.message : 'Unknown error',
        type: 'internal',
        timestamp: new Date().toISOString(),
        service: 'acl',
      };
    }

    return adapter.transformErrors(error);
  }
}

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

interface CanonicalRequest {
  protocol: string;
  method?: string;
  service?: string;
  headers: Record<string, string>;
  body: unknown;
  query?: Record<string, string>;
  path?: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

interface CanonicalResponse {
  protocol: string;
  statusCode?: number;
  statusText?: string;
  headers: Record<string, string>;
  body: unknown;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

type CanonicalData =
  | CanonicalUser
  | CanonicalService
  | CanonicalWorkflow
  | CanonicalKnowledgeDocument
  | CanonicalLogEntry;

interface CanonicalError {
  code: string;
  message: string;
  type: string;
  details?: Record<string, unknown>;
  timestamp: string;
  service: string;
}

// ============================================================================
// EXPORTS
// ============================================================================

export default AntiCorruptionLayer;

export {
  IProtocolAdapter,
  IDataTransformer,
  HttpProtocolAdapter,
  GrpcProtocolAdapter,
  KnowledgeDataTransformer,
  CircuitBreaker,
  CircuitBreakerState,
  type CanonicalRequest,
  type CanonicalResponse,
  type CanonicalData,
  type CanonicalError,
};

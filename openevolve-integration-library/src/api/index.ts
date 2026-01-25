/**
 * OpenEvolve API Client - Main Entry Point
 *
 * Exports all API client functionality
 */

// Main client
export { OpenEvolveClient, createOpenEvolveClient, IntegrationName, IntegrationRegistry } from './client';

// Backend client
export { BackendClient, createBackendClient, WebSocketHandlers } from './backend';

// Types
export type {
  ClientConfig,
  ExecutionOptions,
  ProgressUpdate,
  BatchRequest,
  BatchResult,
  HealthStatus,
  BackendStatus,
  IntegrationHealth,
  WebSocketMessage,
  ConnectionState,
  IntegrationAdapter,
  ValidationResult,
  ValidationErrorItem,
  ValidationWarning,
  RetryConfig,
  RequestMetrics,
  ApiResponse,
  ResponseMeta,
  RequestTransform,
  ResponseTransform,
  ErrorTransform,
} from './types';

// Error classes
export {
  IntegrationError,
  ConnectionError,
  AuthenticationError,
  AuthorizationError,
  ValidationError,
  ExecutionError,
  TimeoutError,
  RateLimitError,
  NotFoundError,
  ConfigurationError,
  NetworkError,
  CancellationError,
  ParseError,
  RetryError,
  isRetryableError,
  isCriticalError,
  createIntegrationError,
} from './errors';

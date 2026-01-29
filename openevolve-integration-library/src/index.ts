/**
 * OpenEvolve Integration Library
 *
 * A generic, reusable library for integrating OpenEvolve components
 * into any frontend application.
 */

// Export API Client and related
export {
  OpenEvolveClient,
  createOpenEvolveClient,
  IntegrationRegistry,
  IntegrationName,
  BackendClient
} from './api/client';

// Export Core Types
export * from './api/types';
export * from './api/errors';

// Export Integrations
export * from './integrations';

// Export Middleware
export * from './api/middleware';

// Export Utilities
export * from './utils/helpers';
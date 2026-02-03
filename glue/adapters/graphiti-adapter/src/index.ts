/**
 * Graphiti Adapter - Main Entry Point
 *
 * Exports all public APIs for the Graphiti temporal knowledge graph adapter.
 *
 * Usage:
 * ```typescript
 * import { GraphitiAdapter } from '@openevolve/graphiti-adapter';
 *
 * const adapter = new GraphitiAdapter({
 *   graphiti_api_url: 'http://localhost:8000',
 *   neo4j_uri: 'bolt://localhost:7687',
 *   neo4j_user: 'neo4j',
 *   neo4j_password: 'password',
 * });
 *
 * await adapter.initialize();
 * const result = await adapter.addEpisode({
 *   name: 'My Episode',
 *   content: 'Some knowledge content',
 *   episode_type: 'text',
 *   valid_at: new Date().toISOString(),
 * });
 * ```
 */

// Main adapter
export { GraphitiAdapter } from './adapter';
export type { GraphitiAdapterConfig } from './adapter';

// Core client
export { GraphitiClient } from './graph-client';
export type { GraphitiClientConfig } from './graph-client';

// Temporal operations
export { GraphitiTemporalOps } from './temporal-ops';

// Version
export const ADAPTER_VERSION = '1.0.0';
export const ADAPTER_NAME = 'graphiti-adapter';

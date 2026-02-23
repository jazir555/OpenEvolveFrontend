/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Adapter - Main Entry Point
 *
 * Exports all public APIs for the ICR adapter.
 */

// Canonical schemas
// Server (started when run directly)
import './server';

export * from './icr-canonical';

// Memory canonical schemas
export * from './memory/canonical';

// ICR Client
export { ICRClient, icrClient } from './icr-client';

// ICR Adapter
export { ICRAdapter, icrAdapter } from './adapter';

// Memory integration
export {
  GraphitiMemoryManager,
  GraphitiMemoryConfig
} from './memory/graphiti-memory';

export {
  EnhancedICRMemoryAgent,
  MemoryAgentConfig
} from './memory/memory-agent';

// Version
export const VERSION = '1.0.0';

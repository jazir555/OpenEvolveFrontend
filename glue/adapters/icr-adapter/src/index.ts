/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Adapter - Main Entry Point
 *
 * Exports all public APIs for the ICR adapter.
 */

// Canonical schemas
export * from './icr-canonical';

// ICR Client
export { ICRClient, icrClient } from './icr-client';

// ICR Adapter
export { ICRAdapter, icrAdapter } from './adapter';

// Version
export const VERSION = '1.0.0';

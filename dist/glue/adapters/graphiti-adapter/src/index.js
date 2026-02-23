"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.ADAPTER_NAME = exports.ADAPTER_VERSION = exports.GraphitiTemporalOps = exports.GraphitiClient = exports.GraphitiAdapter = void 0;
// Main adapter
var adapter_1 = require("./adapter");
Object.defineProperty(exports, "GraphitiAdapter", { enumerable: true, get: function () { return adapter_1.GraphitiAdapter; } });
// Core client
var graph_client_1 = require("./graph-client");
Object.defineProperty(exports, "GraphitiClient", { enumerable: true, get: function () { return graph_client_1.GraphitiClient; } });
// Temporal operations
var temporal_ops_1 = require("./temporal-ops");
Object.defineProperty(exports, "GraphitiTemporalOps", { enumerable: true, get: function () { return temporal_ops_1.GraphitiTemporalOps; } });
// Version
exports.ADAPTER_VERSION = '1.0.0';
exports.ADAPTER_NAME = 'graphiti-adapter';
//# sourceMappingURL=index.js.map
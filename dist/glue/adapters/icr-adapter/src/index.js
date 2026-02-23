"use strict";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Adapter - Main Entry Point
 *
 * Exports all public APIs for the ICR adapter.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.VERSION = exports.EnhancedICRMemoryAgent = exports.GraphitiMemoryManager = exports.icrAdapter = exports.ICRAdapter = exports.icrClient = exports.ICRClient = void 0;
// Canonical schemas
__exportStar(require("./icr-canonical"), exports);
// Memory canonical schemas
__exportStar(require("./memory/canonical"), exports);
// ICR Client
var icr_client_1 = require("./icr-client");
Object.defineProperty(exports, "ICRClient", { enumerable: true, get: function () { return icr_client_1.ICRClient; } });
Object.defineProperty(exports, "icrClient", { enumerable: true, get: function () { return icr_client_1.icrClient; } });
// ICR Adapter
var adapter_1 = require("./adapter");
Object.defineProperty(exports, "ICRAdapter", { enumerable: true, get: function () { return adapter_1.ICRAdapter; } });
Object.defineProperty(exports, "icrAdapter", { enumerable: true, get: function () { return adapter_1.icrAdapter; } });
// Memory integration
var graphiti_memory_1 = require("./memory/graphiti-memory");
Object.defineProperty(exports, "GraphitiMemoryManager", { enumerable: true, get: function () { return graphiti_memory_1.GraphitiMemoryManager; } });
var memory_agent_1 = require("./memory/memory-agent");
Object.defineProperty(exports, "EnhancedICRMemoryAgent", { enumerable: true, get: function () { return memory_agent_1.EnhancedICRMemoryAgent; } });
// Server (started when run directly)
require("./server");
// Version
exports.VERSION = '1.0.0';
//# sourceMappingURL=index.js.map
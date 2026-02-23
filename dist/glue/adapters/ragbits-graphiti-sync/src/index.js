"use strict";
/**
 * RAGBits-Graphiti Bidirectional Sync Adapter
 *
 * Main entry point for the synchronization adapter
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Configuration Explicitness: All config via env vars
 * - Failure Management: Circuit breakers and retries
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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = exports.ConflictDetector = exports.GraphitiToRAGBitsSync = exports.RAGBitsToGraphitiSync = exports.SyncManager = void 0;
const sync_manager_1 = __importDefault(require("./sync-manager"));
exports.SyncManager = sync_manager_1.default;
exports.default = sync_manager_1.default;
const ragbits_to_graphiti_1 = __importDefault(require("./ragbits-to-graphiti"));
exports.RAGBitsToGraphitiSync = ragbits_to_graphiti_1.default;
const graphiti_to_ragbits_1 = __importDefault(require("./graphiti-to-ragbits"));
exports.GraphitiToRAGBitsSync = graphiti_to_ragbits_1.default;
const conflict_detector_1 = __importDefault(require("./conflict-detector"));
exports.ConflictDetector = conflict_detector_1.default;
// Export canonical schemas
__exportStar(require("./canonical"), exports);
//# sourceMappingURL=index.js.map
"use strict";
/**
 * RAGBits Adapter - Exports
 *
 * Following Federation Constitution:
 * - AIR GAP: No imports from core-projects
 * - RUNTIME TRUTH: Verify before use
 * - IDEMPOTENCY: Safe to retry
 * - CIRCUIT BREAKER: Fail fast on dead service
 *
 * @module ragbits-adapter
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RAGExamples = exports.isRAGResponse = exports.isRAGRequest = exports.validateDocumentChunk = exports.validateRAGResponse = exports.validateRAGRequest = exports.transformCanonicalToRAGRequest = exports.transformRAGResponseToCanonical = exports.RAGError = exports.DocumentIngestionResponse = exports.DocumentIngestionRequest = exports.RAGResponse = exports.RAGRequest = exports.DocumentChunk = exports.RAGBitsAdapter = exports.RAGClient = void 0;
var rag_client_1 = require("./rag-client");
Object.defineProperty(exports, "RAGClient", { enumerable: true, get: function () { return rag_client_1.RAGClient; } });
var adapter_1 = require("./adapter");
Object.defineProperty(exports, "RAGBitsAdapter", { enumerable: true, get: function () { return adapter_1.RAGBitsAdapter; } });
// Re-export canonical schemas
var ragbits_canonical_1 = require("../../schemas/ragbits-canonical");
Object.defineProperty(exports, "DocumentChunk", { enumerable: true, get: function () { return ragbits_canonical_1.DocumentChunk; } });
Object.defineProperty(exports, "RAGRequest", { enumerable: true, get: function () { return ragbits_canonical_1.RAGRequest; } });
Object.defineProperty(exports, "RAGResponse", { enumerable: true, get: function () { return ragbits_canonical_1.RAGResponse; } });
Object.defineProperty(exports, "DocumentIngestionRequest", { enumerable: true, get: function () { return ragbits_canonical_1.DocumentIngestionRequest; } });
Object.defineProperty(exports, "DocumentIngestionResponse", { enumerable: true, get: function () { return ragbits_canonical_1.DocumentIngestionResponse; } });
Object.defineProperty(exports, "RAGError", { enumerable: true, get: function () { return ragbits_canonical_1.RAGError; } });
Object.defineProperty(exports, "transformRAGResponseToCanonical", { enumerable: true, get: function () { return ragbits_canonical_1.transformRAGResponseToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToRAGRequest", { enumerable: true, get: function () { return ragbits_canonical_1.transformCanonicalToRAGRequest; } });
Object.defineProperty(exports, "validateRAGRequest", { enumerable: true, get: function () { return ragbits_canonical_1.validateRAGRequest; } });
Object.defineProperty(exports, "validateRAGResponse", { enumerable: true, get: function () { return ragbits_canonical_1.validateRAGResponse; } });
Object.defineProperty(exports, "validateDocumentChunk", { enumerable: true, get: function () { return ragbits_canonical_1.validateDocumentChunk; } });
Object.defineProperty(exports, "isRAGRequest", { enumerable: true, get: function () { return ragbits_canonical_1.isRAGRequest; } });
Object.defineProperty(exports, "isRAGResponse", { enumerable: true, get: function () { return ragbits_canonical_1.isRAGResponse; } });
Object.defineProperty(exports, "RAGExamples", { enumerable: true, get: function () { return ragbits_canonical_1.RAGExamples; } });
//# sourceMappingURL=index.js.map
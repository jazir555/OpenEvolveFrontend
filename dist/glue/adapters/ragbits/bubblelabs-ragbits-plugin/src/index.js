"use strict";
// BubbleLabs RAGBits Plugin - Main Export
// Standalone plugin for semantic document search and knowledge retrieval
Object.defineProperty(exports, "__esModule", { value: true });
exports.ragbitsPlugin = exports.useRAGBitsPlugin = exports.getRAGBitsPlugin = exports.createRAGBitsPlugin = exports.RagbitsService = exports.RagbitsClient = exports.useRAGBitsIngest = exports.useRAGBitsSearch = exports.useRAGBitsState = exports.useRAGBitsConfig = exports.RAGBitsSearchResults = exports.RAGBitsStatusIndicator = exports.RAGBitsIngestPanel = exports.RAGBitsSearchPanel = exports.RAGBitsConfigPanel = exports.DEFAULT_RAGBITS_CONFIG = exports.RAGBITS_DOCUMENT_TYPES = exports.RAGBITS_SEARCH_TYPES = void 0;
exports.createPlugin = createPlugin;
const createRAGBitsPlugin_1 = require("./utils/createRAGBitsPlugin");
var plugin_types_1 = require("./types/plugin-types");
Object.defineProperty(exports, "RAGBITS_SEARCH_TYPES", { enumerable: true, get: function () { return plugin_types_1.RAGBITS_SEARCH_TYPES; } });
Object.defineProperty(exports, "RAGBITS_DOCUMENT_TYPES", { enumerable: true, get: function () { return plugin_types_1.RAGBITS_DOCUMENT_TYPES; } });
Object.defineProperty(exports, "DEFAULT_RAGBITS_CONFIG", { enumerable: true, get: function () { return plugin_types_1.DEFAULT_RAGBITS_CONFIG; } });
// Export components
var RAGBitsConfigPanel_1 = require("./components/RAGBitsConfigPanel");
Object.defineProperty(exports, "RAGBitsConfigPanel", { enumerable: true, get: function () { return RAGBitsConfigPanel_1.RAGBitsConfigPanel; } });
var RAGBitsSearchPanel_1 = require("./components/RAGBitsSearchPanel");
Object.defineProperty(exports, "RAGBitsSearchPanel", { enumerable: true, get: function () { return RAGBitsSearchPanel_1.RAGBitsSearchPanel; } });
var RAGBitsIngestPanel_1 = require("./components/RAGBitsIngestPanel");
Object.defineProperty(exports, "RAGBitsIngestPanel", { enumerable: true, get: function () { return RAGBitsIngestPanel_1.RAGBitsIngestPanel; } });
var RAGBitsStatusIndicator_1 = require("./components/RAGBitsStatusIndicator");
Object.defineProperty(exports, "RAGBitsStatusIndicator", { enumerable: true, get: function () { return RAGBitsStatusIndicator_1.RAGBitsStatusIndicator; } });
var RAGBitsSearchResults_1 = require("./components/RAGBitsSearchResults");
Object.defineProperty(exports, "RAGBitsSearchResults", { enumerable: true, get: function () { return RAGBitsSearchResults_1.RAGBitsSearchResults; } });
// Export hooks
var useRAGBitsConfig_1 = require("./hooks/useRAGBitsConfig");
Object.defineProperty(exports, "useRAGBitsConfig", { enumerable: true, get: function () { return useRAGBitsConfig_1.useRAGBitsConfig; } });
var useRAGBitsState_1 = require("./hooks/useRAGBitsState");
Object.defineProperty(exports, "useRAGBitsState", { enumerable: true, get: function () { return useRAGBitsState_1.useRAGBitsState; } });
var useRAGBitsSearch_1 = require("./hooks/useRAGBitsSearch");
Object.defineProperty(exports, "useRAGBitsSearch", { enumerable: true, get: function () { return useRAGBitsSearch_1.useRAGBitsSearch; } });
var useRAGBitsIngest_1 = require("./hooks/useRAGBitsIngest");
Object.defineProperty(exports, "useRAGBitsIngest", { enumerable: true, get: function () { return useRAGBitsIngest_1.useRAGBitsIngest; } });
// Export services
var ragbitsClient_1 = require("./lib/ragbitsClient");
Object.defineProperty(exports, "RagbitsClient", { enumerable: true, get: function () { return ragbitsClient_1.RagbitsClient; } });
var ragbitsService_1 = require("./services/ragbitsService");
Object.defineProperty(exports, "RagbitsService", { enumerable: true, get: function () { return ragbitsService_1.RagbitsService; } });
// Export utilities
var createRAGBitsPlugin_2 = require("./utils/createRAGBitsPlugin");
Object.defineProperty(exports, "createRAGBitsPlugin", { enumerable: true, get: function () { return createRAGBitsPlugin_2.createRAGBitsPlugin; } });
Object.defineProperty(exports, "getRAGBitsPlugin", { enumerable: true, get: function () { return createRAGBitsPlugin_2.getRAGBitsPlugin; } });
Object.defineProperty(exports, "useRAGBitsPlugin", { enumerable: true, get: function () { return createRAGBitsPlugin_2.useRAGBitsPlugin; } });
/**
 * Create a new RAGBits plugin instance
 * @param config Optional initial configuration
 * @returns RAGBitsPlugin instance
 */
function createPlugin(config) {
    return (0, createRAGBitsPlugin_1.createRAGBitsPlugin)(config);
}
/**
 * Default plugin instance
 */
exports.ragbitsPlugin = (0, createRAGBitsPlugin_1.createRAGBitsPlugin)();
exports.default = exports.ragbitsPlugin;
//# sourceMappingURL=index.js.map
"use strict";
// BubbleLabs Datapizza Plugin - Main Export
// Standalone plugin for data pipeline processing and querying
Object.defineProperty(exports, "__esModule", { value: true });
exports.datapizzaPlugin = exports.useDatapizzaPlugin = exports.createDatapizzaPlugin = exports.DatapizzaService = exports.DatapizzaClient = exports.useDatapizzaQuery = exports.useDatapizzaProcessing = exports.useDatapizzaPipeline = exports.useDatapizzaState = exports.useDatapizzaConfig = exports.DatapizzaPipelinePanel = exports.DatapizzaConfigPanel = exports.DEFAULT_DATAPIZZA_CONFIG = exports.DATAPIZZA_DATA_DOMAINS = exports.DATAPIZZA_PIPELINE_TYPES = void 0;
exports.createPlugin = createPlugin;
var plugin_types_1 = require("./types/plugin-types");
Object.defineProperty(exports, "DATAPIZZA_PIPELINE_TYPES", { enumerable: true, get: function () { return plugin_types_1.DATAPIZZA_PIPELINE_TYPES; } });
Object.defineProperty(exports, "DATAPIZZA_DATA_DOMAINS", { enumerable: true, get: function () { return plugin_types_1.DATAPIZZA_DATA_DOMAINS; } });
Object.defineProperty(exports, "DEFAULT_DATAPIZZA_CONFIG", { enumerable: true, get: function () { return plugin_types_1.DEFAULT_DATAPIZZA_CONFIG; } });
// Export components
var DatapizzaConfigPanel_1 = require("./components/DatapizzaConfigPanel");
Object.defineProperty(exports, "DatapizzaConfigPanel", { enumerable: true, get: function () { return DatapizzaConfigPanel_1.DatapizzaConfigPanel; } });
var DatapizzaPipelinePanel_1 = require("./components/DatapizzaPipelinePanel");
Object.defineProperty(exports, "DatapizzaPipelinePanel", { enumerable: true, get: function () { return DatapizzaPipelinePanel_1.DatapizzaPipelinePanel; } });
// Export hooks (stubs for now)
var useDatapizzaConfig_1 = require("./hooks/useDatapizzaConfig");
Object.defineProperty(exports, "useDatapizzaConfig", { enumerable: true, get: function () { return useDatapizzaConfig_1.useDatapizzaConfig; } });
var useDatapizzaState_1 = require("./hooks/useDatapizzaState");
Object.defineProperty(exports, "useDatapizzaState", { enumerable: true, get: function () { return useDatapizzaState_1.useDatapizzaState; } });
var useDatapizzaPipeline_1 = require("./hooks/useDatapizzaPipeline");
Object.defineProperty(exports, "useDatapizzaPipeline", { enumerable: true, get: function () { return useDatapizzaPipeline_1.useDatapizzaPipeline; } });
var useDatapizzaProcessing_1 = require("./hooks/useDatapizzaProcessing");
Object.defineProperty(exports, "useDatapizzaProcessing", { enumerable: true, get: function () { return useDatapizzaProcessing_1.useDatapizzaProcessing; } });
var useDatapizzaQuery_1 = require("./hooks/useDatapizzaQuery");
Object.defineProperty(exports, "useDatapizzaQuery", { enumerable: true, get: function () { return useDatapizzaQuery_1.useDatapizzaQuery; } });
// Export services (stubs for now)
var DatapizzaClient_1 = require("./services/DatapizzaClient");
Object.defineProperty(exports, "DatapizzaClient", { enumerable: true, get: function () { return DatapizzaClient_1.DatapizzaClient; } });
var DatapizzaService_1 = require("./services/DatapizzaService");
Object.defineProperty(exports, "DatapizzaService", { enumerable: true, get: function () { return DatapizzaService_1.DatapizzaService; } });
// Export utilities
var createDatapizzaPlugin_1 = require("./utils/createDatapizzaPlugin");
Object.defineProperty(exports, "createDatapizzaPlugin", { enumerable: true, get: function () { return createDatapizzaPlugin_1.createDatapizzaPlugin; } });
var createDatapizzaPlugin_2 = require("./utils/createDatapizzaPlugin");
Object.defineProperty(exports, "useDatapizzaPlugin", { enumerable: true, get: function () { return createDatapizzaPlugin_2.useDatapizzaPlugin; } });
const createDatapizzaPlugin_3 = require("./utils/createDatapizzaPlugin");
/**
 * Create a new Datapizza plugin instance
 * @param config Optional initial configuration
 * @returns DatapizzaPlugin instance
 */
function createPlugin(config) {
    return (0, createDatapizzaPlugin_3.createDatapizzaPlugin)(config);
}
/**
 * Default plugin instance
 */
exports.datapizzaPlugin = (0, createDatapizzaPlugin_3.createDatapizzaPlugin)();
exports.default = exports.datapizzaPlugin;
//# sourceMappingURL=index.js.map
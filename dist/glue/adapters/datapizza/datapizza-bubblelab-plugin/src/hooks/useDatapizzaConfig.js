"use strict";
// Datapizza Configuration Hook
// React hook for managing Datapizza plugin configuration
Object.defineProperty(exports, "__esModule", { value: true });
exports.useDatapizzaConfig = useDatapizzaConfig;
const react_1 = require("react");
const plugin_types_1 = require("../types/plugin-types");
function useDatapizzaConfig() {
    const [config, setConfig] = (0, react_1.useState)(plugin_types_1.DEFAULT_DATAPIZZA_CONFIG);
    const updateConfig = (newConfig) => {
        setConfig((prev) => ({ ...prev, ...newConfig }));
    };
    return [config, updateConfig];
}
//# sourceMappingURL=useDatapizzaConfig.js.map
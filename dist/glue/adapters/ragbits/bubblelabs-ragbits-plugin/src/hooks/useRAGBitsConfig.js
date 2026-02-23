"use strict";
// RAGBits Configuration Hook
Object.defineProperty(exports, "__esModule", { value: true });
exports.useRAGBitsConfig = useRAGBitsConfig;
const react_1 = require("react");
const plugin_types_1 = require("../types/plugin-types");
const createRAGBitsPlugin_1 = require("../utils/createRAGBitsPlugin");
function useRAGBitsConfig() {
    const plugin = (0, createRAGBitsPlugin_1.useRAGBitsPlugin)();
    const [config, setConfig] = (0, react_1.useState)({
        ...plugin_types_1.DEFAULT_RAGBITS_CONFIG,
        ...plugin.getContext().config
    });
    const updateConfig = (0, react_1.useCallback)(async (configUpdate) => {
        try {
            await plugin.updateConfig(configUpdate);
            setConfig(prev => ({ ...prev, ...configUpdate }));
        }
        catch (error) {
            console.error('Failed to update config:', error);
            throw error;
        }
    }, [plugin]);
    return [config, updateConfig];
}
//# sourceMappingURL=useRAGBitsConfig.js.map
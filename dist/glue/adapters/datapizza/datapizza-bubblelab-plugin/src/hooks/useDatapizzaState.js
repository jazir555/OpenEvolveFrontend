"use strict";
// Datapizza State Hook
// React hook for accessing Datapizza plugin state
Object.defineProperty(exports, "__esModule", { value: true });
exports.useDatapizzaState = useDatapizzaState;
const react_1 = require("react");
const plugin_types_1 = require("../types/plugin-types");
function useDatapizzaState() {
    const [state] = (0, react_1.useState)({
        ...plugin_types_1.DEFAULT_DATAPIZZA_CONFIG,
        status: 'idle',
        operationHistory: [],
        statistics: {
            totalOperations: 0,
            successfulOperations: 0,
            failedOperations: 0,
            averageProcessingTime: 0
        }
    });
    return state;
}
//# sourceMappingURL=useDatapizzaState.js.map
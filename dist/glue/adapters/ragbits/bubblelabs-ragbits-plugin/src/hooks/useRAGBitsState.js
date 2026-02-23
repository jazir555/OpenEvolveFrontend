"use strict";
// RAGBits State Hook
Object.defineProperty(exports, "__esModule", { value: true });
exports.useRAGBitsState = useRAGBitsState;
const react_1 = require("react");
const createRAGBitsPlugin_1 = require("../utils/createRAGBitsPlugin");
function useRAGBitsState() {
    const plugin = (0, createRAGBitsPlugin_1.useRAGBitsPlugin)();
    const [state, setState] = (0, react_1.useState)(plugin.getContext().state);
    (0, react_1.useEffect)(() => {
        const interval = setInterval(() => {
            setState(plugin.getContext().state);
        }, 1000);
        return () => clearInterval(interval);
    }, [plugin]);
    return state;
}
//# sourceMappingURL=useRAGBitsState.js.map
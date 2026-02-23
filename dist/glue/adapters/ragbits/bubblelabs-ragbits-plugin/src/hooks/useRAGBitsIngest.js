"use strict";
// RAGBits Ingest Hook
Object.defineProperty(exports, "__esModule", { value: true });
exports.useRAGBitsIngest = useRAGBitsIngest;
const react_1 = require("react");
const createRAGBitsPlugin_1 = require("../utils/createRAGBitsPlugin");
function useRAGBitsIngest() {
    const plugin = (0, createRAGBitsPlugin_1.useRAGBitsPlugin)();
    const ingest = (0, react_1.useCallback)(async (request) => {
        return await plugin.ingest(request);
    }, [plugin]);
    return ingest;
}
//# sourceMappingURL=useRAGBitsIngest.js.map
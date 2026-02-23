"use strict";
// RAGBits Search Hook
Object.defineProperty(exports, "__esModule", { value: true });
exports.useRAGBitsSearch = useRAGBitsSearch;
const react_1 = require("react");
const createRAGBitsPlugin_1 = require("../utils/createRAGBitsPlugin");
function useRAGBitsSearch() {
    const plugin = (0, createRAGBitsPlugin_1.useRAGBitsPlugin)();
    const search = (0, react_1.useCallback)(async (request) => {
        return await plugin.search(request);
    }, [plugin]);
    return search;
}
//# sourceMappingURL=useRAGBitsSearch.js.map
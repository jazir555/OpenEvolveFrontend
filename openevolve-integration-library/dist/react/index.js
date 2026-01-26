"use strict";
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
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.useOpenEvolveClient = exports.OpenEvolveProvider = void 0;
exports.useIntegration = useIntegration;
exports.useDecomposition = useDecomposition;
exports.useLeanAide = useLeanAide;
exports.useEvolution = useEvolution;
exports.useKnowledgeEngine = useKnowledgeEngine;
exports.useMaker = useMaker;
exports.useHephaestus = useHephaestus;
exports.useSolution = useSolution;
exports.useVerification = useVerification;
exports.useAssembly = useAssembly;
const react_1 = __importStar(require("react"));
const client_1 = require("../api/client");
const errors_1 = require("../api/errors");
const OpenEvolveContext = (0, react_1.createContext)(null);
const OpenEvolveProvider = ({ client, children }) => {
    const value = react_1.default.useMemo(() => client, [client]);
    return react_1.default.createElement(OpenEvolveContext.Provider, { value }, children);
};
exports.OpenEvolveProvider = OpenEvolveProvider;
const useOpenEvolveClient = () => {
    const client = (0, react_1.useContext)(OpenEvolveContext);
    if (!client) {
        throw new Error('useOpenEvolveClient must be used within an OpenEvolveProvider');
    }
    return client;
};
exports.useOpenEvolveClient = useOpenEvolveClient;
function useIntegration(integrationName) {
    const client = (0, exports.useOpenEvolveClient)();
    const [data, setData] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [error, setError] = (0, react_1.useState)(null);
    const abortControllerRef = (0, react_1.useRef)(null);
    const executionCountRef = (0, react_1.useRef)(0);
    (0, react_1.useEffect)(() => {
        return () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);
    const reset = (0, react_1.useCallback)(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        setData(null);
        setLoading(false);
        setError(null);
    }, []);
    const execute = (0, react_1.useCallback)(async (inputs) => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        const executionId = ++executionCountRef.current;
        const controller = new AbortController();
        abortControllerRef.current = controller;
        setLoading(true);
        setError(null);
        try {
            const result = await client.execute(integrationName, inputs, { signal: controller.signal });
            if (executionId === executionCountRef.current) {
                setData(result);
                setLoading(false);
                return result;
            }
            return null;
        }
        catch (err) {
            if (err.name === 'AbortError' || err.code === 'CANCELLATION_ERROR' || (err instanceof errors_1.IntegrationError && err.code === 'CANCELLATION_ERROR')) {
                if (executionId === executionCountRef.current) {
                    setLoading(false);
                }
                return null;
            }
            const integrationError = (0, errors_1.createIntegrationError)(integrationName, err);
            if (executionId === executionCountRef.current) {
                setError(integrationError);
                setLoading(false);
                throw integrationError;
            }
            return null;
        }
    }, [client, integrationName]);
    return { data, loading, error, execute, reset };
}
function useDecomposition() {
    return useIntegration(client_1.IntegrationName.DECOMPOSITION);
}
function useLeanAide() {
    return useIntegration(client_1.IntegrationName.LEANAIDE);
}
function useEvolution() {
    return useIntegration(client_1.IntegrationName.EVOLUTION);
}
function useKnowledgeEngine() {
    return useIntegration(client_1.IntegrationName.KNOWLEDGE);
}
function useMaker() {
    return useIntegration(client_1.IntegrationName.MAKER);
}
function useHephaestus() {
    return useIntegration(client_1.IntegrationName.HEPHAESTUS);
}
function useSolution() {
    return useIntegration(client_1.IntegrationName.SOLUTION);
}
function useVerification() {
    return useIntegration(client_1.IntegrationName.VERIFICATION);
}
function useAssembly() {
    return useIntegration(client_1.IntegrationName.ASSEMBLY);
}
//# sourceMappingURL=index.js.map
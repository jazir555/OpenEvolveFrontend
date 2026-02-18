/**
 * OpenEvolve Plugin Constants
 */
export declare const API_ENDPOINTS: {
    readonly BASE: "/api/openevolve";
    readonly WEBSOCKET: "/ws/openevolve";
    readonly WORKFLOWS: "/api/openevolve/workflows";
    readonly ANALYTICS: "/api/openevolve/analytics";
    readonly KNOWLEDGE: "/api/openevolve/knowledge";
    readonly LEANAIDE: "/api/openevolve/leanaide";
    readonly EVOLUTION: "/api/openevolve/evolution";
    readonly ADVERSARIAL: "/api/openevolve/adversarial";
    readonly MAKER: "/api/openevolve/maker";
    readonly MDAP: "/api/openevolve/mdap";
    readonly DECOMPOSITION: "/api/openevolve/decomposition";
    readonly CREWAI: "/api/openevolve/crewai";
    readonly ROMA: "/api/openevolve/roma";
    readonly INVENTION: "/api/openevolve/invention";
};
export declare const WORKFLOW_TYPES: {
    readonly EVOLUTION: "evolution";
    readonly ADVERSARIAL: "adversarial";
    readonly MAKER: "maker";
    readonly MDAP: "mdap";
    readonly DECOMPOSITION: "decomposition";
    readonly INVENTION: "invention";
};
export declare const EXECUTION_STATUS: {
    readonly IDLE: "idle";
    readonly RUNNING: "running";
    readonly COMPLETED: "completed";
    readonly FAILED: "failed";
};
export declare const ARTIFACT_TYPES: {
    readonly MODEL: "model";
    readonly DATASET: "dataset";
    readonly PROOF: "proof";
    readonly WORKFLOW: "workflow";
    readonly RESULT: "result";
    readonly LOG: "log";
};
export declare const LEAN_MODELS: {
    readonly MATHLIB: "mathlib";
    readonly STD: "std";
    readonly ALEAN: "alean";
    readonly COUNTEREXAMPLES: "counterexamples";
};
export declare const PROOF_STATUS: {
    readonly PENDING: "pending";
    readonly PROVING: "proving";
    readonly VERIFIED: "verified";
    readonly FAILED: "failed";
};
export declare const THEME_COLORS: {
    readonly PRIMARY: "#3b82f6";
    readonly SUCCESS: "#10b981";
    readonly WARNING: "#f59e0b";
    readonly ERROR: "#ef4444";
    readonly INFO: "#6366f1";
};
export declare const CHART_COLORS: readonly ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#6366f1", "#8b5cf6", "#ec4899", "#14b8a6"];
export declare const DEFAULT_PAGINATION: {
    readonly PAGE: 1;
    readonly PAGE_SIZE: 20;
};
export declare const WEBSOCKET_EVENTS: {
    readonly CONNECT: "connect";
    readonly DISCONNECT: "disconnect";
    readonly ERROR: "error";
    readonly WORKFLOW_STARTED: "workflow.started";
    readonly WORKFLOW_UPDATED: "workflow.updated";
    readonly WORKFLOW_COMPLETED: "workflow.completed";
    readonly WORKFLOW_FAILED: "workflow.failed";
    readonly LOG_MESSAGE: "log.message";
    readonly PROOF_PROGRESS: "proof.progress";
    readonly ANALYTICS_UPDATE: "analytics.update";
};
export declare const LOCAL_STORAGE_KEYS: {
    readonly AUTH_TOKEN: "openevolve_auth_token";
    readonly USER_PREFERENCES: "openevolve_preferences";
    readonly RECENT_WORKFLOWS: "openevolve_recent_workflows";
    readonly SAVED_QUERIES: "openevolve_saved_queries";
};

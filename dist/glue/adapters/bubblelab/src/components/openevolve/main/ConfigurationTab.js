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
exports.ConfigurationTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const DEFAULT_CONFIG = {
    max_iterations: 100,
    population_size: 1000,
    num_islands: 5,
    migration_interval: 50,
    migration_rate: 0.1,
    archive_size: 100,
    elite_ratio: 0.1,
    exploration_ratio: 0.2,
    exploitation_ratio: 0.7,
    checkpoint_interval: 100,
    temperature: 0.7,
    top_p: 0.95,
    max_tokens: 4096,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    feature_dimensions: ["complexity", "diversity"],
    feature_bins: 10,
    diversity_metric: "edit_distance",
    enable_artifacts: true,
    cascade_evaluation: true,
    cascade_thresholds: [0.5, 0.75, 0.9],
    use_llm_feedback: false,
    llm_feedback_weight: 0.1,
    parallel_evaluations: 4,
    distributed: false,
    template_dir: null,
    num_top_programs: 3,
    num_diverse_programs: 2,
    use_template_stochasticity: true,
    template_variations: {},
    use_meta_prompting: false,
    meta_prompt_weight: 0.1,
    include_artifacts: true,
    max_artifact_bytes: 20480,
    artifact_security_filter: true,
    early_stopping_patience: null,
    convergence_threshold: 0.001,
    early_stopping_metric: "combined_score",
    memory_limit_mb: null,
    cpu_limit: null,
    random_seed: 42,
    db_path: null,
    in_memory: true,
    diff_based_evolution: true,
    max_code_length: 10000,
    evolution_trace_enabled: false,
    evolution_trace_format: "jsonl",
    evolution_trace_include_code: false,
    evolution_trace_include_prompts: true,
    evolution_trace_output_path: null,
    evolution_trace_buffer_size: 10,
    evolution_trace_compress: false,
    log_level: "INFO",
    log_dir: null,
    api_timeout: 60,
    api_retries: 3,
    api_retry_delay: 5,
    artifact_size_threshold: 32768,
    cleanup_old_artifacts: true,
    artifact_retention_days: 30,
    diversity_reference_size: 20,
    max_retries_eval: 3,
    evaluator_timeout: 300,
    double_selection: true,
    adaptive_feature_dimensions: true,
    test_time_compute: false,
    optillm_integration: false,
    plugin_system: false,
    hardware_optimization: false,
    multi_strategy_sampling: true,
    ring_topology: true,
    controlled_gene_flow: true,
    auto_diff: true,
    symbolic_execution: false,
    coevolutionary_approach: false,
};
const PRESETS = {
    default: DEFAULT_CONFIG,
    research: {
        ...DEFAULT_CONFIG,
        num_islands: 7,
        population_size: 2000,
        archive_size: 200,
        use_llm_feedback: true,
        llm_feedback_weight: 0.15,
        parallel_evaluations: 8,
        evolution_trace_enabled: true,
        evolution_trace_include_code: true,
        early_stopping_patience: 20,
        convergence_threshold: 0.0001,
        double_selection: true,
        adaptive_feature_dimensions: true,
        multi_strategy_sampling: true,
        test_time_compute: true,
    },
    production: {
        ...DEFAULT_CONFIG,
        num_islands: 3,
        population_size: 500,
        archive_size: 50,
        parallel_evaluations: 2,
        use_llm_feedback: false,
        memory_limit_mb: 2048,
        cpu_limit: 2.0,
        evolution_trace_enabled: false,
        early_stopping_patience: 10,
        convergence_threshold: 0.001,
        api_timeout: 30,
        api_retries: 2,
    },
    experimental: {
        ...DEFAULT_CONFIG,
        num_islands: 10,
        population_size: 3000,
        archive_size: 300,
        use_llm_feedback: true,
        llm_feedback_weight: 0.2,
        parallel_evaluations: 8,
        evolution_trace_enabled: true,
        evolution_trace_include_prompts: true,
        evolution_trace_include_code: true,
        early_stopping_patience: 30,
        convergence_threshold: 0.00001,
        double_selection: true,
        adaptive_feature_dimensions: true,
        test_time_compute: true,
        optillm_integration: true,
        plugin_system: true,
        hardware_optimization: true,
        multi_strategy_sampling: true,
        symbolic_execution: true,
        coevolutionary_approach: true,
    },
};
const readConfig = () => {
    try {
        const raw = globalThis.localStorage?.getItem("openevolve_config");
        if (raw)
            return JSON.parse(raw);
    }
    catch {
        // ignore
    }
    return DEFAULT_CONFIG;
};
const saveConfig = (config) => {
    try {
        globalThis.localStorage?.setItem("openevolve_config", JSON.stringify(config));
    }
    catch {
        // ignore
    }
};
const downloadJson = (filename, payload) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};
const ConfigurationTab = () => {
    const [preset, setPreset] = (0, react_1.useState)("default");
    const [config, setConfig] = (0, react_1.useState)(readConfig);
    const [configDraft, setConfigDraft] = (0, react_1.useState)(() => JSON.stringify(config, null, 2));
    const [message, setMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const summary = (0, react_1.useMemo)(() => {
        return {
            max_iterations: config.max_iterations ?? "n/a",
            population_size: config.population_size ?? "n/a",
            num_islands: config.num_islands ?? "n/a",
            temperature: config.temperature ?? "n/a",
            use_llm_feedback: config.use_llm_feedback ?? false,
            cascade_evaluation: config.cascade_evaluation ?? false,
        };
    }, [config]);
    const applyPreset = () => {
        const next = PRESETS[preset] ?? DEFAULT_CONFIG;
        setConfig(next);
        setConfigDraft(JSON.stringify(next, null, 2));
        saveConfig(next);
        setMessage(`Applied preset: ${preset}`);
        setErrorMessage(null);
    };
    const handleSave = () => {
        try {
            const parsed = JSON.parse(configDraft);
            setConfig(parsed);
            saveConfig(parsed);
            setMessage("Configuration saved.");
            setErrorMessage(null);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Invalid JSON.");
            setMessage(null);
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Configuration System</card_1.CardTitle>
          <card_1.CardDescription>Manage OpenEvolve configuration presets and overrides.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-[240px_1fr]">
            <div className="space-y-2">
              <label_1.Label>Preset</label_1.Label>
              <select className="w-full rounded border border-input bg-background px-3 py-2 text-sm" value={preset} onChange={(event) => setPreset(event.target.value)}>
                <option value="default">Default</option>
                <option value="research">Research</option>
                <option value="production">Production</option>
                <option value="experimental">Experimental</option>
              </select>
              <button_1.Button variant="outline" onClick={applyPreset}>
                Apply Preset
              </button_1.Button>
            </div>
            <div className="space-y-2">
              <label_1.Label>Configuration JSON</label_1.Label>
              <textarea_1.Textarea value={configDraft} onChange={(event) => setConfigDraft(event.target.value)} rows={12}/>
              <div className="flex flex-wrap gap-2">
                <button_1.Button onClick={handleSave}>Save Configuration</button_1.Button>
                <button_1.Button variant="outline" onClick={() => downloadJson("openevolve_config.json", config)}>
                  Download JSON
                </button_1.Button>
              </div>
            </div>
          </div>

          {message ? <div className="text-sm text-green-600">{message}</div> : null}
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <separator_1.Separator />
          <div className="grid gap-3 md:grid-cols-3 text-sm">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-xs">Core Settings</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1">
                <div>Max Iterations: {String(summary.max_iterations)}</div>
                <div>Population Size: {String(summary.population_size)}</div>
                <div>Islands: {String(summary.num_islands)}</div>
                <div>Temperature: {String(summary.temperature)}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-xs">Advanced Features</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1">
                <div>LLM Feedback: {summary.use_llm_feedback ? "Enabled" : "Disabled"}</div>
                <div>Cascade Evaluation: {summary.cascade_evaluation ? "Enabled" : "Disabled"}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-xs">Preset Health</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2">
                <badge_1.Badge variant="secondary">Active Preset: {preset}</badge_1.Badge>
                <div className="text-xs text-muted-foreground">
                  Update JSON to fine-tune experimental parameters.
                </div>
              </card_1.CardContent>
            </card_1.Card>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ConfigurationTab = ConfigurationTab;
//# sourceMappingURL=ConfigurationTab.js.map
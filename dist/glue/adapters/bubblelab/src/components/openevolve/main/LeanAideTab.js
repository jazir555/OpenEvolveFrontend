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
exports.LeanAideTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const tabs_1 = require("@/components/ui/tabs");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readApiKey = () => {
    try {
        return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    }
    catch {
        return "";
    }
};
const LeanAideTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(readApiKey);
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [status, setStatus] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [lastResult, setLastResult] = (0, react_1.useState)(null);
    const [theoremText, setTheoremText] = (0, react_1.useState)("");
    const [theoremName, setTheoremName] = (0, react_1.useState)("");
    const [theoremCode, setTheoremCode] = (0, react_1.useState)("");
    const [mathQuery, setMathQuery] = (0, react_1.useState)("");
    const [mctsConfig, setMctsConfig] = (0, react_1.useState)({
        max_iterations: 1000,
        time_budget: 300,
        c_param: 1.414,
        expansion_agents: 3,
        simulation_voters: 5,
    });
    const [treeIds, setTreeIds] = (0, react_1.useState)([]);
    const [selectedTreeId, setSelectedTreeId] = (0, react_1.useState)("");
    const [treeDetail, setTreeDetail] = (0, react_1.useState)(null);
    const [proofIds, setProofIds] = (0, react_1.useState)([]);
    const [selectedProofId, setSelectedProofId] = (0, react_1.useState)("");
    const [proofDetail, setProofDetail] = (0, react_1.useState)(null);
    const loadStatus = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsLeanAideStatus(apiConfig);
            setStatus(response);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load LeanAide status.");
        }
        finally {
            setLoading(false);
        }
    };
    const loadTrees = async () => {
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsLeanAideTrees(apiConfig);
            setTreeIds(response.tree_ids ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load trees.");
        }
    };
    const loadProofs = async () => {
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsLeanAideProofs(apiConfig);
            setProofIds(response.proof_ids ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load proofs.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadStatus();
        loadTrees();
        loadProofs();
    }, [apiConfig.apiKey]);
    const executeTask = async (taskType, payload) => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsLeanAideExecute({ task_type: taskType, payload }, apiConfig);
            setLastResult(response.result ?? null);
            await loadTrees();
            await loadProofs();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "LeanAide task failed.");
        }
        finally {
            setLoading(false);
        }
    };
    const handleSelectTree = async (treeId) => {
        setSelectedTreeId(treeId);
        if (!treeId) {
            setTreeDetail(null);
            return;
        }
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsLeanAideTree(treeId, apiConfig);
            setTreeDetail(response);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load tree.");
        }
    };
    const handleSelectProof = async (proofId) => {
        setSelectedProofId(proofId);
        if (!proofId) {
            setProofDetail(null);
            return;
        }
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsLeanAideProof(proofId, apiConfig);
            setProofDetail(response);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load proof.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>LeanAide Formal Verification</card_1.CardTitle>
          <card_1.CardDescription>Run theorem proving, MCTS search, and Lean4 verification.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input type="password" value={apiKey} onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_key", value);
            }
            catch {
                // ignore storage errors
            }
        }}/>
            </div>
            <button_1.Button variant="outline" onClick={loadStatus} disabled={loading}>
              Refresh Status
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          {status ? (<div className="grid gap-4 md:grid-cols-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">MCTS</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="text-2xl font-semibold">
                  {status.mcts_available ? "✅" : "❌"}
                </card_1.CardContent>
              </card_1.Card>
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">MDAP</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="text-2xl font-semibold">
                  {status.mdap_available ? "✅" : "❌"}
                </card_1.CardContent>
              </card_1.Card>
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Lean4</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="text-2xl font-semibold">
                  {status.lean4_available ? "✅" : "❌"}
                </card_1.CardContent>
              </card_1.Card>
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Active Proofs</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="text-2xl font-semibold">{status.active_proofs}</card_1.CardContent>
              </card_1.Card>
            </div>) : null}

          <tabs_1.Tabs defaultValue="theorem" className="w-full">
            <tabs_1.TabsList className="flex flex-wrap gap-2">
              <tabs_1.TabsTrigger value="theorem">Theorem Proving</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="mcts">MCTS Visualization</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="lean4">Lean4 Verification</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="math">Math Queries</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="theorem" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Theorem Proving</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label_1.Label>Theorem Statement</label_1.Label>
                    <textarea_1.Textarea value={theoremText} onChange={(event) => setTheoremText(event.target.value)} className="min-h-[120px]"/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Theorem Name</label_1.Label>
                    <input_1.Input value={theoremName} onChange={(event) => setTheoremName(event.target.value)} placeholder="Optional name"/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Theorem Code (optional Lean)</label_1.Label>
                    <textarea_1.Textarea value={theoremCode} onChange={(event) => setTheoremCode(event.target.value)} className="min-h-[120px]"/>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button_1.Button variant="outline" onClick={() => executeTask("translate_theorem", {
            theorem_text: theoremText,
            theorem_name: theoremName || undefined,
        })} disabled={loading}>
                      Translate
                    </button_1.Button>
                    <button_1.Button variant="outline" onClick={() => executeTask("generate_proof", {
            theorem_text: theoremText,
            theorem_code: theoremCode || undefined,
        })} disabled={loading}>
                      Generate Proof
                    </button_1.Button>
                    <button_1.Button variant="outline" onClick={() => executeTask("verify_solution", {
            code: theoremCode,
        })} disabled={loading}>
                      Verify Code
                    </button_1.Button>
                    <button_1.Button variant="outline" onClick={() => executeTask("mcts_search", {
            theorem: theoremText,
            theorem_name: theoremName || undefined,
            ...mctsConfig,
        })} disabled={loading}>
                      MCTS Search
                    </button_1.Button>
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="mcts" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">MCTS Trees</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Select Tree</label_1.Label>
                      <select_1.Select value={selectedTreeId} onValueChange={handleSelectTree}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="Select tree"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {treeIds.map((treeId) => (<select_1.SelectItem key={treeId} value={treeId}>
                              {treeId.slice(0, 8)}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Iterations</label_1.Label>
                      <input_1.Input type="number" value={mctsConfig.max_iterations} onChange={(event) => setMctsConfig((prev) => ({
            ...prev,
            max_iterations: Number(event.target.value) || 0,
        }))}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Time Budget (s)</label_1.Label>
                      <input_1.Input type="number" value={mctsConfig.time_budget} onChange={(event) => setMctsConfig((prev) => ({
            ...prev,
            time_budget: Number(event.target.value) || 0,
        }))}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Exploration Constant</label_1.Label>
                      <input_1.Input type="number" value={mctsConfig.c_param} onChange={(event) => setMctsConfig((prev) => ({
            ...prev,
            c_param: Number(event.target.value) || 0,
        }))}/>
                    </div>
                  </div>

                  {treeDetail ? (<div className="rounded border p-3 text-xs whitespace-pre-wrap">
                      {JSON.stringify(treeDetail.tree, null, 2)}
                    </div>) : (<div className="text-sm text-muted-foreground">Select a tree to view details.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="lean4" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Lean4 Proofs</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label_1.Label>Select Proof</label_1.Label>
                    <select_1.Select value={selectedProofId} onValueChange={handleSelectProof}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Select proof"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {proofIds.map((proofId) => (<select_1.SelectItem key={proofId} value={proofId}>
                            {proofId.slice(0, 8)}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                  {proofDetail ? (<div className="rounded border p-3 text-xs whitespace-pre-wrap">
                      {JSON.stringify(proofDetail.proof, null, 2)}
                    </div>) : (<div className="text-sm text-muted-foreground">Select a proof to view details.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="math" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Math Query</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label_1.Label>Query</label_1.Label>
                    <textarea_1.Textarea value={mathQuery} onChange={(event) => setMathQuery(event.target.value)} className="min-h-[120px]"/>
                  </div>
                  <button_1.Button variant="outline" onClick={() => executeTask("math_query", { query: mathQuery })} disabled={loading}>
                    Run Query
                  </button_1.Button>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>

          {lastResult ? (<card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-base">Last Result</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent>
                <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(lastResult, null, 2)}</pre>
              </card_1.CardContent>
            </card_1.Card>) : null}
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.LeanAideTab = LeanAideTab;
//# sourceMappingURL=LeanAideTab.js.map
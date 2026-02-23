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
exports.BubbleLabsIntegrationTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const parseJson = (value) => {
    if (!value.trim())
        return undefined;
    return JSON.parse(value);
};
const controlPayloadTemplate = (component, action) => {
    if (component !== "openevolve_workflows") {
        return {};
    }
    switch (action) {
        case "create_definition":
            return {
                name: "OpenEvolve Workflow",
                description: "Managed from BubbleLabs control tab",
                workflow_type: "evolution",
                parameters: {},
            };
        case "get_definition":
            return { definition_id: "" };
        case "create_instance":
            return { definition_id: "", instance_name: "bubble-instance", inputs: {}, parameters: {} };
        case "get_instance_status":
            return { instance_id: "" };
        case "start_instance":
        case "pause_instance":
        case "resume_instance":
        case "stop_instance":
        case "cancel_instance":
        case "restart_instance":
        case "delete_instance":
            return { instance_id: "" };
        case "sync_parameters":
            return { instance_id: "", parameters: {} };
        default:
            return {};
    }
};
const BubbleLabsIntegrationTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [status, setStatus] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [aceSkillbookName, setAceSkillbookName] = (0, react_1.useState)("");
    const [aceSkills, setAceSkills] = (0, react_1.useState)("[]");
    const [acePatterns, setAcePatterns] = (0, react_1.useState)("[]");
    const [z3Variables, setZ3Variables] = (0, react_1.useState)("[]");
    const [z3Constraints, setZ3Constraints] = (0, react_1.useState)("");
    const [z3Theorem, setZ3Theorem] = (0, react_1.useState)("");
    const [romaProblem, setRomaProblem] = (0, react_1.useState)("");
    const [romaDepth, setRomaDepth] = (0, react_1.useState)("3");
    const [romaConfig, setRomaConfig] = (0, react_1.useState)("{}");
    const [knowledgeArtifact, setKnowledgeArtifact] = (0, react_1.useState)("{}");
    const [knowledgeQuery, setKnowledgeQuery] = (0, react_1.useState)("");
    const [analyticsWorkflowId, setAnalyticsWorkflowId] = (0, react_1.useState)("");
    const [analyticsMetrics, setAnalyticsMetrics] = (0, react_1.useState)("{}");
    const [analyticsDashboard, setAnalyticsDashboard] = (0, react_1.useState)(null);
    const [leanTheorem, setLeanTheorem] = (0, react_1.useState)("");
    const [actionResult, setActionResult] = (0, react_1.useState)(null);
    const [controlCatalog, setControlCatalog] = (0, react_1.useState)(null);
    const [controlComponent, setControlComponent] = (0, react_1.useState)("");
    const [controlAction, setControlAction] = (0, react_1.useState)("");
    const [controlPayload, setControlPayload] = (0, react_1.useState)("{}");
    const controlComponents = (0, react_1.useMemo)(() => Object.keys(controlCatalog?.components ?? {}).sort(), [controlCatalog]);
    const controlActions = (0, react_1.useMemo)(() => {
        if (!controlComponent || !controlCatalog?.components[controlComponent]) {
            return [];
        }
        return [...controlCatalog.components[controlComponent]].sort();
    }, [controlCatalog, controlComponent]);
    const refreshStatus = async () => {
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getBubblelabsStatus(apiConfig);
            setStatus(response);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load BubbleLabs status.");
        }
    };
    const refreshControlCatalog = async () => {
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsControlCatalog(apiConfig);
            setControlCatalog(response);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load BubbleLabs control catalog.");
        }
    };
    (0, react_1.useEffect)(() => {
        refreshStatus();
        refreshControlCatalog();
    }, [apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        if (!controlComponents.length) {
            setControlComponent("");
            return;
        }
        if (!controlComponents.includes(controlComponent)) {
            setControlComponent(controlComponents[0]);
        }
    }, [controlComponent, controlComponents]);
    (0, react_1.useEffect)(() => {
        if (!controlActions.length) {
            setControlAction("");
            setControlPayload("{}");
            return;
        }
        if (!controlActions.includes(controlAction)) {
            const nextAction = controlActions[0];
            setControlAction(nextAction);
            setControlPayload(JSON.stringify(controlPayloadTemplate(controlComponent, nextAction), null, 2));
        }
    }, [controlAction, controlActions, controlComponent]);
    const runAction = async (fn) => {
        setErrorMessage(null);
        setStatusMessage(null);
        try {
            const response = await fn();
            setActionResult(response);
            setStatusMessage("Action completed.");
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Action failed.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>BubbleLabs Integration</card_1.CardTitle>
          <card_1.CardDescription>Manage extended integrations (ACE, Z3, ROMA, LeanAide).</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_key", value);
            }
            catch {
                // ignore
            }
        }}/>
            </div>
            <div className="flex gap-2">
              <button_1.Button variant="outline" onClick={refreshStatus}>
                Refresh Status
              </button_1.Button>
              <button_1.Button variant="outline" onClick={() => runAction(async () => {
            const response = await openevolveApi_1.openevolveApi.initializeBubblelabs(apiConfig);
            await refreshStatus();
            return response;
        })}>
                Initialize
              </button_1.Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-3 md:grid-cols-3 text-sm">
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Components</div>
              <div className="text-lg font-semibold">{status?.total_components ?? 0}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Available</div>
              <div className="text-lg font-semibold">{status?.available_components ?? 0}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Health</div>
              <div className="text-lg font-semibold">
                {status
            ? `${Math.round(((status.available_components ?? 0) / (status.total_components || 1)) * 100)}%`
            : "n/a"}
              </div>
            </div>
          </div>

          {status?.components ? (<div className="grid gap-3 md:grid-cols-2">
              {Object.entries(status.components).map(([key, component]) => (<card_1.Card key={key}>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">{String(component.component ?? key)}</card_1.CardTitle>
                    <card_1.CardDescription>{String(component.status ?? "unknown")}</card_1.CardDescription>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-xs">
                    <div>Version: {String(component.version ?? "n/a")}</div>
                    {Array.isArray(component.capabilities) && component.capabilities.length ? (<div className="flex flex-wrap gap-2">
                        {component.capabilities.map((cap) => (<badge_1.Badge key={cap} variant="secondary">
                            {cap}
                          </badge_1.Badge>))}
                      </div>) : null}
                  </card_1.CardContent>
                </card_1.Card>))}
            </div>) : null}
        </card_1.CardContent>
      </card_1.Card>

      <tabs_1.Tabs defaultValue="workflows" className="w-full">
        <tabs_1.TabsList className="grid w-full grid-cols-8">
          <tabs_1.TabsTrigger value="workflows">Workflows</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="control">Control</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="ace">ACE</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="z3">Z3</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="roma">ROMA</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="knowledge">Knowledge</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="analytics">Analytics</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="lean">LeanAide</tabs_1.TabsTrigger>
        </tabs_1.TabsList>

        <tabs_1.TabsContent value="workflows" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Workflow Templates</card_1.CardTitle>
              <card_1.CardDescription>
                Pre-built workflows that combine multiple BubbleLabs capabilities
              </card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded border p-3">
                  <h4 className="font-medium">Research Assistant</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Search knowledge base, analyze results, and generate insights
                  </p>
                  <badge_1.Badge variant="secondary">RAGBits + Datapizza</badge_1.Badge>
                </div>
                <div className="rounded border p-3">
                  <h4 className="font-medium">Data Analysis Pipeline</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Process raw data through ETL pipeline and generate analytics
                  </p>
                  <badge_1.Badge variant="secondary">Datapizza + RAGBits</badge_1.Badge>
                </div>
                <div className="rounded border p-3">
                  <h4 className="font-medium">Proof Verification</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Verify mathematical theorems using Z3 and LeanAide
                  </p>
                  <badge_1.Badge variant="secondary">Z3 + LeanAide</badge_1.Badge>
                </div>
                <div className="rounded border p-3">
                  <h4 className="font-medium">Knowledge Extraction</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Extract structured knowledge from unstructured documents
                  </p>
                  <badge_1.Badge variant="secondary">Knowledge + RAGBits</badge_1.Badge>
                </div>
                <div className="rounded border p-3">
                  <h4 className="font-medium">Problem Solving</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Analyze complex problems and generate solutions
                  </p>
                  <badge_1.Badge variant="secondary">ROMA + Z3</badge_1.Badge>
                </div>
              </div>
              <button_1.Button onClick={() => {
            // Navigate to workflow execution tab
            const tabElement = document.querySelector('[value="workflow-execution"]');
            tabElement?.click();
        }}>
                Open Workflow Executor
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="control" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Unified Control Plane</card_1.CardTitle>
              <card_1.CardDescription>
                Dynamically execute discovered BubbleLabs and OpenEvolve integration actions.
              </card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <button_1.Button variant="outline" onClick={refreshControlCatalog}>
                  Refresh Catalog
                </button_1.Button>
                <button_1.Button variant="outline" onClick={() => runAction(async () => {
            const response = await openevolveApi_1.openevolveApi.bubblelabsControlDiscover({ force: true }, apiConfig);
            await refreshControlCatalog();
            return response;
        })}>
                  Discover Integrations
                </button_1.Button>
                <badge_1.Badge variant="secondary">
                  Components: {controlComponents.length}
                </badge_1.Badge>
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <label_1.Label>Component</label_1.Label>
                  <select value={controlComponent} onChange={(event) => setControlComponent(event.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                    {controlComponents.length === 0 ? (<option value="">No components available</option>) : null}
                    {controlComponents.map((component) => (<option key={component} value={component}>
                        {component}
                      </option>))}
                  </select>
                </div>

                <div className="space-y-2">
                  <label_1.Label>Action</label_1.Label>
                  <select value={controlAction} onChange={(event) => {
            const nextAction = event.target.value;
            setControlAction(nextAction);
            setControlPayload(JSON.stringify(controlPayloadTemplate(controlComponent, nextAction), null, 2));
        }} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                    {controlActions.length === 0 ? (<option value="">No actions available</option>) : null}
                    {controlActions.map((action) => (<option key={action} value={action}>
                        {action}
                      </option>))}
                  </select>
                </div>
              </div>

              <div className="space-y-2">
                <label_1.Label>Payload (JSON)</label_1.Label>
                <textarea_1.Textarea value={controlPayload} onChange={(event) => setControlPayload(event.target.value)} rows={7}/>
              </div>

              <button_1.Button onClick={() => runAction(async () => {
            const parsed = parseJson(controlPayload) ?? {};
            if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
                throw new Error("Payload must be a JSON object");
            }
            return await openevolveApi_1.openevolveApi.bubblelabsControlExecute({ component: controlComponent, action: controlAction, payload: parsed }, apiConfig);
        })} disabled={!controlComponent || !controlAction}>
                Execute Control Action
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="ace" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Create Skillbook</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <input_1.Input placeholder="Skillbook name" value={aceSkillbookName} onChange={(event) => setAceSkillbookName(event.target.value)}/>
              <textarea_1.Textarea value={aceSkills} onChange={(event) => setAceSkills(event.target.value)} rows={4}/>
              <button_1.Button onClick={() => runAction(async () => openevolveApi_1.openevolveApi.bubblelabsAceSkillbook({ name: aceSkillbookName, skills: parseJson(aceSkills) ?? [] }, apiConfig))}>
                Create Skillbook
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Extract Patterns</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <textarea_1.Textarea value={acePatterns} onChange={(event) => setAcePatterns(event.target.value)} rows={4}/>
              <button_1.Button onClick={() => runAction(async () => openevolveApi_1.openevolveApi.bubblelabsAcePatterns({ workflow_results: parseJson(acePatterns) ?? [] }, apiConfig))}>
                Extract Patterns
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="z3" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Solve Constraints</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <textarea_1.Textarea value={z3Variables} onChange={(event) => setZ3Variables(event.target.value)} rows={3} placeholder='[{"name": "x", "type": "Int"}]'/>
              <textarea_1.Textarea value={z3Constraints} onChange={(event) => setZ3Constraints(event.target.value)} rows={3} placeholder="(> x 0)\n(< x 10)"/>
              <button_1.Button onClick={() => runAction(async () => openevolveApi_1.openevolveApi.bubblelabsZ3Solve({
            variables: parseJson(z3Variables) ?? [],
            constraints: z3Constraints.split("\n").filter(Boolean),
        }, apiConfig))}>
                Solve
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Prove Theorem</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <input_1.Input value={z3Theorem} onChange={(event) => setZ3Theorem(event.target.value)} placeholder="forall x. x > 0"/>
              <button_1.Button onClick={() => runAction(() => openevolveApi_1.openevolveApi.bubblelabsZ3Prove({ theorem: z3Theorem }, apiConfig))}>
                Prove
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="roma" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Analyze Problem</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <textarea_1.Textarea value={romaProblem} onChange={(event) => setRomaProblem(event.target.value)} rows={3}/>
              <input_1.Input value={romaDepth} onChange={(event) => setRomaDepth(event.target.value)} placeholder="Max depth"/>
              <button_1.Button onClick={() => runAction(() => openevolveApi_1.openevolveApi.bubblelabsRomaAnalyze({ problem: romaProblem, max_depth: Number(romaDepth) }, apiConfig))}>
                Analyze
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Create Config</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <textarea_1.Textarea value={romaConfig} onChange={(event) => setRomaConfig(event.target.value)} rows={4}/>
              <button_1.Button onClick={() => runAction(() => openevolveApi_1.openevolveApi.bubblelabsRomaConfig({ config: parseJson(romaConfig) ?? {} }, apiConfig))}>
                Create Config
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="knowledge" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Store Artifact</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <textarea_1.Textarea value={knowledgeArtifact} onChange={(event) => setKnowledgeArtifact(event.target.value)} rows={4}/>
              <button_1.Button onClick={() => runAction(() => openevolveApi_1.openevolveApi.bubblelabsKnowledgeStore({ artifact: parseJson(knowledgeArtifact) ?? {} }, apiConfig))}>
                Store Artifact
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Query Patterns</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <input_1.Input value={knowledgeQuery} onChange={(event) => setKnowledgeQuery(event.target.value)}/>
              <button_1.Button onClick={() => runAction(() => openevolveApi_1.openevolveApi.bubblelabsKnowledgeQuery({ query: knowledgeQuery }, apiConfig))}>
                Query
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="analytics" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Track Workflow Metrics</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <input_1.Input value={analyticsWorkflowId} onChange={(event) => setAnalyticsWorkflowId(event.target.value)} placeholder="Workflow ID"/>
              <textarea_1.Textarea value={analyticsMetrics} onChange={(event) => setAnalyticsMetrics(event.target.value)} rows={4}/>
              <button_1.Button onClick={() => runAction(() => openevolveApi_1.openevolveApi.bubblelabsAnalyticsTrack({ workflow_id: analyticsWorkflowId, metrics: parseJson(analyticsMetrics) ?? {} }, apiConfig))}>
                Track Metrics
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Analytics Dashboard</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <button_1.Button onClick={() => runAction(async () => {
            const response = await openevolveApi_1.openevolveApi.bubblelabsAnalyticsDashboard(apiConfig);
            setAnalyticsDashboard(response);
            return response;
        })}>
                Load Dashboard
              </button_1.Button>
              {analyticsDashboard ? (<pre className="rounded border p-2 text-xs whitespace-pre-wrap">
                  {JSON.stringify(analyticsDashboard, null, 2)}
                </pre>) : null}
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="lean" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">LeanAide Prover</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <input_1.Input value={leanTheorem} onChange={(event) => setLeanTheorem(event.target.value)} placeholder="Theorem to prove"/>
              <button_1.Button onClick={() => runAction(() => openevolveApi_1.openevolveApi.bubblelabsLeanAideProve({ theorem: leanTheorem }, apiConfig))}>
                Prove
              </button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>
      </tabs_1.Tabs>

      {actionResult ? (<card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-sm">Action Result</card_1.CardTitle>
          </card_1.CardHeader>
          <card_1.CardContent>
            <pre className="rounded border p-2 text-xs whitespace-pre-wrap">
              {JSON.stringify(actionResult, null, 2)}
            </pre>
          </card_1.CardContent>
        </card_1.Card>) : null}
    </div>);
};
exports.BubbleLabsIntegrationTab = BubbleLabsIntegrationTab;
//# sourceMappingURL=BubbleLabsIntegrationTab.js.map
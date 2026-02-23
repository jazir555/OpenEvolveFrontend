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
exports.SovereignDashboardTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const textarea_1 = require("@/components/ui/textarea");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const SovereignDashboardTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [health, setHealth] = (0, react_1.useState)(null);
    const [problems, setProblems] = (0, react_1.useState)([]);
    const [plans, setPlans] = (0, react_1.useState)([]);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const refresh = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [healthRes, problemsRes, plansRes] = await Promise.all([
                openevolveApi_1.openevolveApi.getSovereignHealth(apiConfig),
                openevolveApi_1.openevolveApi.listSovereignProblems(apiConfig),
                openevolveApi_1.openevolveApi.listSovereignPlans(apiConfig),
            ]);
            setHealth(healthRes);
            setProblems(problemsRes.problems ?? []);
            setPlans(plansRes.plans ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load sovereign dashboard data.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        refresh();
    }, [apiConfig.apiKey]);
    const activePlans = plans.filter((plan) => plan.status === "ACTIVE" || plan.status === "in_execution");
    const planStrategies = plans.reduce((acc, plan) => {
        const strategy = String(plan.strategy ?? "unknown");
        acc[strategy] = (acc[strategy] || 0) + 1;
        return acc;
    }, {});
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Sovereign Decomposition Dashboard</card_1.CardTitle>
          <card_1.CardDescription>Monitor problem decomposition and orchestration health.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
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
            <button_1.Button variant="outline" onClick={refresh} disabled={loading}>
              Refresh
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-4 text-sm">
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Problems</div>
              <div className="text-lg font-semibold">{problems.length}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Active Plans</div>
              <div className="text-lg font-semibold">{activePlans.length}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Health</div>
              <div className="text-lg font-semibold">
                {String(health?.overall_status ?? "unknown")}
              </div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Uptime</div>
              <div className="text-lg font-semibold">{String(health?.uptime ?? "n/a")}</div>
            </div>
          </div>

          <tabs_1.Tabs defaultValue="problems">
            <tabs_1.TabsList className="grid w-full grid-cols-4">
              <tabs_1.TabsTrigger value="problems">Problems</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="plans">Plans</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="health">Health</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="analytics">Analytics</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="problems" className="mt-4 space-y-3">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Problem Definitions</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {problems.length === 0 && (<div className="text-muted-foreground">No problems available.</div>)}
                  {problems.map((problem) => (<div key={String(problem.id)} className="rounded border p-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{String(problem.title ?? problem.id)}</div>
                        <badge_1.Badge variant="secondary">{String(problem.problem_type ?? "unknown")}</badge_1.Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Domain: {String(problem.domain_context?.domain ?? "n/a")} · Status:{" "}
                        {String(problem.status ?? "n/a")}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Created: {String(problem.created_at ?? "n/a")}
                      </div>
                    </div>))}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="plans" className="mt-4 space-y-3">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Decomposition Plans</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {plans.length === 0 && (<div className="text-muted-foreground">No plans available.</div>)}
                  {plans.map((plan) => (<div key={String(plan.id)} className="rounded border p-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">Plan {String(plan.id)}</div>
                        <badge_1.Badge variant="secondary">{String(plan.status ?? "unknown")}</badge_1.Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Strategy: {String(plan.strategy ?? "n/a")} · Sub-problems:{" "}
                        {Array.isArray(plan.sub_problems) ? plan.sub_problems.length : 0}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Confidence: {String(plan.confidence_level ?? "n/a")}
                      </div>
                    </div>))}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="health" className="mt-4 space-y-3">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">System Health</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent>
                  <textarea_1.Textarea value={health ? JSON.stringify(health, null, 2) : ""} readOnly className="min-h-[200px]"/>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="analytics" className="mt-4 space-y-3">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Plan Strategy Distribution</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {Object.entries(planStrategies).map(([strategy, count]) => (<div key={strategy} className="flex items-center justify-between rounded border p-2">
                      <span>{strategy}</span>
                      <badge_1.Badge variant="secondary">{count}</badge_1.Badge>
                    </div>))}
                  {Object.keys(planStrategies).length === 0 && (<div className="text-muted-foreground">No analytics available.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.SovereignDashboardTab = SovereignDashboardTab;
//# sourceMappingURL=SovereignDashboardTab.js.map
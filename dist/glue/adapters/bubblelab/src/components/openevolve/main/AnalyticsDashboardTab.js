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
exports.AnalyticsDashboardTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const AnalyticsDashboardTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [statistics, setStatistics] = (0, react_1.useState)(null);
    const [overview, setOverview] = (0, react_1.useState)(null);
    const [components, setComponents] = (0, react_1.useState)(null);
    const [refinements, setRefinements] = (0, react_1.useState)(null);
    const [workflowMetrics, setWorkflowMetrics] = (0, react_1.useState)([]);
    const [teamMetrics, setTeamMetrics] = (0, react_1.useState)([]);
    const [gauntletMetrics, setGauntletMetrics] = (0, react_1.useState)([]);
    const [knowledgeStats, setKnowledgeStats] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const refresh = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [stats, icrOverview, icrComponents, icrRefinements, workflowPerf, teamPerf, gauntletPerf, knowledgeSummary,] = await Promise.all([
                openevolveApi_1.openevolveApi.getStatistics(apiConfig),
                openevolveApi_1.openevolveApi.getIcrOverview(apiConfig),
                openevolveApi_1.openevolveApi.getIcrComponents(apiConfig),
                openevolveApi_1.openevolveApi.getIcrRefinements(apiConfig),
                openevolveApi_1.openevolveApi.getPerformanceMetrics("workflow", 200, apiConfig),
                openevolveApi_1.openevolveApi.getPerformanceMetrics("team", 200, apiConfig),
                openevolveApi_1.openevolveApi.getPerformanceMetrics("gauntlet", 200, apiConfig),
                openevolveApi_1.openevolveApi.getAnalyticsKnowledgeStats(apiConfig),
            ]);
            setStatistics(stats);
            setOverview(icrOverview);
            setComponents(icrComponents);
            setRefinements(icrRefinements);
            setWorkflowMetrics(workflowPerf.metrics || []);
            setTeamMetrics(teamPerf.metrics || []);
            setGauntletMetrics(gauntletPerf.metrics || []);
            setKnowledgeStats(knowledgeSummary);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load analytics.");
        }
        finally {
            setLoading(false);
        }
    };
    const getMetricValue = (metric, keys) => {
        for (const key of keys) {
            const value = metric.metrics?.[key];
            if (typeof value === "number") {
                return value;
            }
        }
        return null;
    };
    const summarizeTeamMetrics = (metrics) => {
        const summary = {};
        metrics.forEach((metric) => {
            const key = metric.entity_id || "unknown";
            if (!summary[key]) {
                summary[key] = { total: 0, success: 0, failure: 0, scores: [] };
            }
            summary[key].total += 1;
            const successValue = getMetricValue(metric, ["success", "passed", "approved"]);
            if (successValue !== null) {
                if (successValue > 0) {
                    summary[key].success += 1;
                }
                else {
                    summary[key].failure += 1;
                }
            }
            const scoreValue = getMetricValue(metric, ["score", "overall_score", "quality_score"]);
            if (scoreValue !== null) {
                summary[key].scores.push(scoreValue);
            }
        });
        return summary;
    };
    const teamSummary = summarizeTeamMetrics(teamMetrics);
    const gauntletSummary = summarizeTeamMetrics(gauntletMetrics);
    const workflowQualityScores = workflowMetrics
        .map((metric) => getMetricValue(metric, ["overall_score", "quality_score", "score", "success_rate"]))
        .filter((value) => typeof value === "number");
    const topWorkflowScores = [...workflowQualityScores].sort((a, b) => b - a).slice(0, 5);
    (0, react_1.useEffect)(() => {
        refresh();
    }, [apiConfig.apiKey]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Analytics Dashboard</card_1.CardTitle>
          <card_1.CardDescription>System-wide metrics and ICR analytics.</card_1.CardDescription>
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
              Refresh Analytics
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-3">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Workflows</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>Total: {statistics?.total_workflows ?? 0}</div>
                <div>Running: {statistics?.running ?? 0}</div>
                <div>Completed: {statistics?.completed ?? 0}</div>
                <div>Failed: {statistics?.failed ?? 0}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Teams & Gauntlets</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>Teams: {statistics?.total_teams ?? 0}</div>
                <div>Gauntlets: {statistics?.total_gauntlets ?? 0}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">ICR Overview</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>
                  Status:{" "}
                  <badge_1.Badge variant={overview?.icr_enabled ? "default" : "secondary"}>
                    {overview?.icr_enabled ? "Enabled" : "Disabled"}
                  </badge_1.Badge>
                </div>
                <div>Total Patterns: {overview?.total_patterns ?? 0}</div>
                <div>Active Components: {overview?.active_components ?? 0}</div>
                <div>Total Refinements: {overview?.total_refinements ?? 0}</div>
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">ICR Components</card_1.CardTitle>
                <card_1.CardDescription>Pass rates and activity by component.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {!components && <div className="text-muted-foreground">No component data.</div>}
                {components &&
            Object.entries(components).map(([name, data]) => (<div key={name} className="rounded border p-2">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{name}</div>
                        <badge_1.Badge variant={data.active ? "default" : "secondary"}>
                          {data.active ? "active" : "inactive"}
                        </badge_1.Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Patterns: {data.total_patterns ?? 0} · Pass rate:{" "}
                        {((data.overall_pass_rate ?? 0) * 100).toFixed(1)}%
                      </div>
                    </div>))}
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Recent Refinements</card_1.CardTitle>
                <card_1.CardDescription>Latest ICR refinement events.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {refinements?.events?.length ? (refinements.events.map((event, index) => (<div key={index} className="rounded border p-2">
                      <div className="font-semibold">{event.refinement_type ?? "Refinement"}</div>
                      <div className="text-xs text-muted-foreground">
                        {event.component ?? "component"} · {event.timestamp ?? "time"}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Success: {String(event.success ?? false)} · Confidence:{" "}
                        {event.confidence ?? 0}
                      </div>
                    </div>))) : (<div className="text-muted-foreground">No refinements recorded.</div>)}
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Workflow Performance</card_1.CardTitle>
                <card_1.CardDescription>Recent workflow metrics and outcomes.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {workflowMetrics.length === 0 ? (<div className="text-muted-foreground">No workflow metrics available.</div>) : (workflowMetrics.slice(-8).reverse().map((metric, index) => (<div key={`${metric.entity_id}-${index}`} className="rounded border p-2">
                      <div className="font-semibold">{metric.entity_id}</div>
                      <div className="text-xs text-muted-foreground">
                        Duration:{" "}
                        {getMetricValue(metric, ["duration_minutes", "duration", "elapsed_minutes"]) ??
                "n/a"}{" "}
                        · Sub-problems:{" "}
                        {getMetricValue(metric, ["sub_problems_solved", "subproblem_count"]) ?? "n/a"} ·
                        Refinements:{" "}
                        {getMetricValue(metric, ["refinement_loops", "refinements"]) ?? "n/a"}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Success:{" "}
                        {getMetricValue(metric, ["success", "passed", "approved"]) ?? "n/a"} ·{" "}
                        {metric.timestamp ?? "timestamp"}
                      </div>
                    </div>)))}
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Solution Quality</card_1.CardTitle>
                <card_1.CardDescription>Top quality scores from recent workflows.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {topWorkflowScores.length === 0 ? (<div className="text-muted-foreground">No quality metrics recorded.</div>) : (topWorkflowScores.map((score, index) => (<div key={`score-${index}`} className="flex items-center justify-between">
                      <span>Score #{index + 1}</span>
                      <badge_1.Badge variant="secondary">{score.toFixed(2)}</badge_1.Badge>
                    </div>)))}
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Team Analytics</card_1.CardTitle>
                <card_1.CardDescription>Aggregated performance by team.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {Object.keys(teamSummary).length === 0 ? (<div className="text-muted-foreground">No team metrics available.</div>) : (Object.entries(teamSummary).map(([teamName, data]) => {
            const avgScore = data.scores.length > 0
                ? data.scores.reduce((a, b) => a + b, 0) / data.scores.length
                : null;
            return (<div key={teamName} className="rounded border p-2">
                        <div className="font-semibold">{teamName}</div>
                        <div className="text-xs text-muted-foreground">
                          Tasks: {data.total} · Successes: {data.success} · Failures: {data.failure}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Avg Score: {avgScore !== null ? avgScore.toFixed(2) : "n/a"}
                        </div>
                      </div>);
        }))}
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Gauntlet Analytics</card_1.CardTitle>
                <card_1.CardDescription>Performance metrics by gauntlet.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {Object.keys(gauntletSummary).length === 0 ? (<div className="text-muted-foreground">No gauntlet metrics available.</div>) : (Object.entries(gauntletSummary).map(([gauntletName, data]) => {
            const avgScore = data.scores.length > 0
                ? data.scores.reduce((a, b) => a + b, 0) / data.scores.length
                : null;
            return (<div key={gauntletName} className="rounded border p-2">
                        <div className="font-semibold">{gauntletName}</div>
                        <div className="text-xs text-muted-foreground">
                          Runs: {data.total} · Pass: {data.success} · Fail: {data.failure}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Avg Score: {avgScore !== null ? avgScore.toFixed(2) : "n/a"}
                        </div>
                      </div>);
        }))}
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Knowledge Base Statistics</card_1.CardTitle>
              <card_1.CardDescription>Artifact coverage and usage trends.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-2 text-sm">
              {!knowledgeStats ? (<div className="text-muted-foreground">Knowledge stats not available.</div>) : (<div className="space-y-3">
                  <div>Total Artifacts: {knowledgeStats.total_artifacts}</div>
                  <div>Total Usage: {knowledgeStats.total_usage}</div>
                  <div>Avg Effectiveness: {knowledgeStats.avg_effectiveness.toFixed(2)}</div>
                  <div>
                    Artifact Types:{" "}
                    {Object.entries(knowledgeStats.artifact_type_distribution)
                .map(([key, value]) => `${key} (${value})`)
                .join(", ")}
                  </div>
                  <div>
                    Domains:{" "}
                    {Object.entries(knowledgeStats.domain_distribution)
                .map(([key, value]) => `${key} (${value})`)
                .join(", ")}
                  </div>
                </div>)}
            </card_1.CardContent>
          </card_1.Card>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.AnalyticsDashboardTab = AnalyticsDashboardTab;
//# sourceMappingURL=AnalyticsDashboardTab.js.map
import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  StatisticsSummary,
  IcrOverview,
  IcrComponents,
  IcrRefinements,
  PerformanceMetric,
  AnalyticsKnowledgeStats,
} from "@/lib/types";

export const AnalyticsDashboardTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [statistics, setStatistics] = useState<StatisticsSummary | null>(null);
  const [overview, setOverview] = useState<IcrOverview | null>(null);
  const [components, setComponents] = useState<IcrComponents | null>(null);
  const [refinements, setRefinements] = useState<IcrRefinements | null>(null);
  const [workflowMetrics, setWorkflowMetrics] = useState<PerformanceMetric[]>([]);
  const [teamMetrics, setTeamMetrics] = useState<PerformanceMetric[]>([]);
  const [gauntletMetrics, setGauntletMetrics] = useState<PerformanceMetric[]>([]);
  const [knowledgeStats, setKnowledgeStats] = useState<AnalyticsKnowledgeStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [
        stats,
        icrOverview,
        icrComponents,
        icrRefinements,
        workflowPerf,
        teamPerf,
        gauntletPerf,
        knowledgeSummary,
      ] = await Promise.all([
        openevolveApi.getStatistics(apiConfig),
        openevolveApi.getIcrOverview(apiConfig),
        openevolveApi.getIcrComponents(apiConfig),
        openevolveApi.getIcrRefinements(apiConfig),
        openevolveApi.getPerformanceMetrics("workflow", 200, apiConfig),
        openevolveApi.getPerformanceMetrics("team", 200, apiConfig),
        openevolveApi.getPerformanceMetrics("gauntlet", 200, apiConfig),
        openevolveApi.getAnalyticsKnowledgeStats(apiConfig),
      ]);
      setStatistics(stats);
      setOverview(icrOverview);
      setComponents(icrComponents);
      setRefinements(icrRefinements);
      setWorkflowMetrics(workflowPerf.metrics || []);
      setTeamMetrics(teamPerf.metrics || []);
      setGauntletMetrics(gauntletPerf.metrics || []);
      setKnowledgeStats(knowledgeSummary);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load analytics.");
    } finally {
      setLoading(false);
    }
  };

  const getMetricValue = (metric: PerformanceMetric, keys: string[]) => {
    for (const key of keys) {
      const value = metric.metrics?.[key];
      if (typeof value === "number") {
        return value;
      }
    }
    return null;
  };

  const summarizeTeamMetrics = (metrics: PerformanceMetric[]) => {
    const summary: Record<
      string,
      { total: number; success: number; failure: number; scores: number[] }
    > = {};
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
        } else {
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
    .map((metric) =>
      getMetricValue(metric, ["overall_score", "quality_score", "score", "success_rate"]),
    )
    .filter((value): value is number => typeof value === "number");

  const topWorkflowScores = [...workflowQualityScores].sort((a, b) => b - a).slice(0, 5);

  useEffect(() => {
    refresh();
  }, [apiConfig.apiKey]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Analytics Dashboard</CardTitle>
          <CardDescription>System-wide metrics and ICR analytics.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
              <Input
                value={apiKey}
                type="password"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  try {
                    globalThis.localStorage?.setItem("openevolve_api_key", value);
                  } catch {
                    // ignore storage errors
                  }
                }}
              />
            </div>
            <Button variant="outline" onClick={refresh} disabled={loading}>
              Refresh Analytics
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Workflows</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>Total: {statistics?.total_workflows ?? 0}</div>
                <div>Running: {statistics?.running ?? 0}</div>
                <div>Completed: {statistics?.completed ?? 0}</div>
                <div>Failed: {statistics?.failed ?? 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Teams & Gauntlets</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>Teams: {statistics?.total_teams ?? 0}</div>
                <div>Gauntlets: {statistics?.total_gauntlets ?? 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">ICR Overview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>
                  Status:{" "}
                  <Badge variant={overview?.icr_enabled ? "default" : "secondary"}>
                    {overview?.icr_enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
                <div>Total Patterns: {overview?.total_patterns ?? 0}</div>
                <div>Active Components: {overview?.active_components ?? 0}</div>
                <div>Total Refinements: {overview?.total_refinements ?? 0}</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">ICR Components</CardTitle>
                <CardDescription>Pass rates and activity by component.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {!components && <div className="text-muted-foreground">No component data.</div>}
                {components &&
                  Object.entries(components).map(([name, data]) => (
                    <div key={name} className="rounded border p-2">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{name}</div>
                        <Badge variant={data.active ? "default" : "secondary"}>
                          {data.active ? "active" : "inactive"}
                        </Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Patterns: {data.total_patterns ?? 0} · Pass rate:{" "}
                        {((data.overall_pass_rate ?? 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Recent Refinements</CardTitle>
                <CardDescription>Latest ICR refinement events.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {refinements?.events?.length ? (
                  refinements.events.map((event, index) => (
                    <div key={index} className="rounded border p-2">
                      <div className="font-semibold">{event.refinement_type ?? "Refinement"}</div>
                      <div className="text-xs text-muted-foreground">
                        {event.component ?? "component"} · {event.timestamp ?? "time"}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Success: {String(event.success ?? false)} · Confidence:{" "}
                        {event.confidence ?? 0}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground">No refinements recorded.</div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Workflow Performance</CardTitle>
                <CardDescription>Recent workflow metrics and outcomes.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {workflowMetrics.length === 0 ? (
                  <div className="text-muted-foreground">No workflow metrics available.</div>
                ) : (
                  workflowMetrics.slice(-8).reverse().map((metric, index) => (
                    <div key={`${metric.entity_id}-${index}`} className="rounded border p-2">
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
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Solution Quality</CardTitle>
                <CardDescription>Top quality scores from recent workflows.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {topWorkflowScores.length === 0 ? (
                  <div className="text-muted-foreground">No quality metrics recorded.</div>
                ) : (
                  topWorkflowScores.map((score, index) => (
                    <div key={`score-${index}`} className="flex items-center justify-between">
                      <span>Score #{index + 1}</span>
                      <Badge variant="secondary">{score.toFixed(2)}</Badge>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Team Analytics</CardTitle>
                <CardDescription>Aggregated performance by team.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {Object.keys(teamSummary).length === 0 ? (
                  <div className="text-muted-foreground">No team metrics available.</div>
                ) : (
                  Object.entries(teamSummary).map(([teamName, data]) => {
                    const avgScore =
                      data.scores.length > 0
                        ? data.scores.reduce((a, b) => a + b, 0) / data.scores.length
                        : null;
                    return (
                      <div key={teamName} className="rounded border p-2">
                        <div className="font-semibold">{teamName}</div>
                        <div className="text-xs text-muted-foreground">
                          Tasks: {data.total} · Successes: {data.success} · Failures: {data.failure}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Avg Score: {avgScore !== null ? avgScore.toFixed(2) : "n/a"}
                        </div>
                      </div>
                    );
                  })
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Gauntlet Analytics</CardTitle>
                <CardDescription>Performance metrics by gauntlet.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {Object.keys(gauntletSummary).length === 0 ? (
                  <div className="text-muted-foreground">No gauntlet metrics available.</div>
                ) : (
                  Object.entries(gauntletSummary).map(([gauntletName, data]) => {
                    const avgScore =
                      data.scores.length > 0
                        ? data.scores.reduce((a, b) => a + b, 0) / data.scores.length
                        : null;
                    return (
                      <div key={gauntletName} className="rounded border p-2">
                        <div className="font-semibold">{gauntletName}</div>
                        <div className="text-xs text-muted-foreground">
                          Runs: {data.total} · Pass: {data.success} · Fail: {data.failure}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Avg Score: {avgScore !== null ? avgScore.toFixed(2) : "n/a"}
                        </div>
                      </div>
                    );
                  })
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Knowledge Base Statistics</CardTitle>
              <CardDescription>Artifact coverage and usage trends.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              {!knowledgeStats ? (
                <div className="text-muted-foreground">Knowledge stats not available.</div>
              ) : (
                <div className="space-y-3">
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
                </div>
              )}
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};

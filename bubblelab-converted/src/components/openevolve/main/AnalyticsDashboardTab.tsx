import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { openevolveApi } from "@/lib/openevolveApi";
import type { StatisticsSummary, IcrOverview, IcrComponents, IcrRefinements } from "@/lib/types";

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
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [stats, icrOverview, icrComponents, icrRefinements] = await Promise.all([
        openevolveApi.getStatistics(apiConfig),
        openevolveApi.getIcrOverview(apiConfig),
        openevolveApi.getIcrComponents(apiConfig),
        openevolveApi.getIcrRefinements(apiConfig),
      ]);
      setStatistics(stats);
      setOverview(icrOverview);
      setComponents(icrComponents);
      setRefinements(icrRefinements);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load analytics.");
    } finally {
      setLoading(false);
    }
  };

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
        </CardContent>
      </Card>
    </div>
  );
};

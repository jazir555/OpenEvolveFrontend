import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { openevolveApi } from "../../../lib/openevolveApi";

export const SovereignDashboardTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [problems, setProblems] = useState<Record<string, unknown>[]>([]);
  const [plans, setPlans] = useState<Record<string, unknown>[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [healthRes, problemsRes, plansRes] = await Promise.all([
        openevolveApi.getSovereignHealth(apiConfig),
        openevolveApi.listSovereignProblems(apiConfig),
        openevolveApi.listSovereignPlans(apiConfig),
      ]);
      setHealth(healthRes);
      setProblems(problemsRes.problems ?? []);
      setPlans(plansRes.plans ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load sovereign dashboard data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, [apiConfig.apiKey]);

  const activePlans = plans.filter((plan) => plan.status === "ACTIVE" || plan.status === "in_execution");
  const planStrategies = plans.reduce<Record<string, number>>((acc, plan) => {
    const strategy = String(plan.strategy ?? "unknown");
    acc[strategy] = (acc[strategy] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Sovereign Decomposition Dashboard</CardTitle>
          <CardDescription>Monitor problem decomposition and orchestration health.</CardDescription>
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
              Refresh
            </Button>
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

          <Tabs defaultValue="problems">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="problems">Problems</TabsTrigger>
              <TabsTrigger value="plans">Plans</TabsTrigger>
              <TabsTrigger value="health">Health</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
            </TabsList>

            <TabsContent value="problems" className="mt-4 space-y-3">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Problem Definitions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {problems.length === 0 && (
                    <div className="text-muted-foreground">No problems available.</div>
                  )}
                  {problems.map((problem) => (
                    <div key={String(problem.id)} className="rounded border p-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{String(problem.title ?? problem.id)}</div>
                        <Badge variant="secondary">{String(problem.problem_type ?? "unknown")}</Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Domain: {String(problem.domain_context?.domain ?? "n/a")} · Status:{" "}
                        {String(problem.status ?? "n/a")}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Created: {String(problem.created_at ?? "n/a")}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="plans" className="mt-4 space-y-3">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Decomposition Plans</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {plans.length === 0 && (
                    <div className="text-muted-foreground">No plans available.</div>
                  )}
                  {plans.map((plan) => (
                    <div key={String(plan.id)} className="rounded border p-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">Plan {String(plan.id)}</div>
                        <Badge variant="secondary">{String(plan.status ?? "unknown")}</Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Strategy: {String(plan.strategy ?? "n/a")} · Sub-problems:{" "}
                        {Array.isArray(plan.sub_problems) ? plan.sub_problems.length : 0}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Confidence: {String(plan.confidence_level ?? "n/a")}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="health" className="mt-4 space-y-3">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">System Health</CardTitle>
                </CardHeader>
                <CardContent>
                  <Textarea
                    value={health ? JSON.stringify(health, null, 2) : ""}
                    readOnly
                    className="min-h-[200px]"
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analytics" className="mt-4 space-y-3">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Plan Strategy Distribution</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {Object.entries(planStrategies).map(([strategy, count]) => (
                    <div key={strategy} className="flex items-center justify-between rounded border p-2">
                      <span>{strategy}</span>
                      <Badge variant="secondary">{count}</Badge>
                    </div>
                  ))}
                  {Object.keys(planStrategies).length === 0 && (
                    <div className="text-muted-foreground">No analytics available.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { AuditLogEntry } from "../../../lib/types";

export const ActivityFeedTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadLogs = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const result = await openevolveApi.listAuditLogs(200, apiConfig);
      setLogs(result.logs ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load audit logs.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLogs();
  }, [apiConfig.apiKey]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Activity Feed</CardTitle>
          <CardDescription>Audit log events across workflows, teams, and gauntlets.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key (admin required)</label>
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
            <Button variant="outline" onClick={loadLogs} disabled={loading}>
              Refresh Logs
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="space-y-3">
            {logs.length === 0 && (
              <div className="text-sm text-muted-foreground">No audit events available.</div>
            )}
            {logs.map((log, index) => (
              <div key={`${log.resource_id ?? "log"}-${index}`} className="rounded border p-3">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <div className="text-sm font-semibold">{log.operation ?? "Event"}</div>
                    <div className="text-xs text-muted-foreground">
                      {log.timestamp ?? "Unknown time"} · {log.user ?? "system"}
                    </div>
                  </div>
                  <Badge variant={log.success ? "default" : "destructive"}>
                    {log.success ? "success" : "failure"}
                  </Badge>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  {log.resource ?? "resource"} · {log.resource_id ?? "n/a"}
                </div>
                {log.details ? (
                  <pre className="mt-2 overflow-auto rounded bg-muted p-2 text-xs">
                    {JSON.stringify(log.details, null, 2)}
                  </pre>
                ) : null}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

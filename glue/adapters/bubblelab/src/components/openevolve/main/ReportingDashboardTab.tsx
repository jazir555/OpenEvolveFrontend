import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";

interface EvolutionReport {
  run_id: string;
  timestamp?: string;
  evolution_mode?: string;
  content_type?: string;
  parameters?: Record<string, unknown>;
  results?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  summary_statistics?: Record<string, unknown>;
}

const readJson = <T,>(key: string, fallback: T): T => {
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

const saveJson = (key: string, value: unknown) => {
  try {
    globalThis.localStorage?.setItem(key, JSON.stringify(value));
  } catch {
    // ignore
  }
};

const downloadJson = (filename: string, payload: unknown) => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

export const ReportingDashboardTab: React.FC = () => {
  const [reports, setReports] = useState<EvolutionReport[]>(() =>
    readJson("openevolve_reports", []),
  );
  const [selectedReport, setSelectedReport] = useState<EvolutionReport | null>(null);
  const [importPayload, setImportPayload] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const [settings, setSettings] = useState(() =>
    readJson("openevolve_reporting_settings", {
      auto_generate_reports: true,
      detailed_visualizations: true,
      preferred_export_formats: ["PDF", "JSON"],
      report_retention: "3 months",
      default_report_template: "Standard Report",
    }),
  );

  const persistReports = (next: EvolutionReport[]) => {
    setReports(next);
    saveJson("openevolve_reports", next);
  };

  const handleImport = () => {
    setErrorMessage(null);
    try {
      const parsed = JSON.parse(importPayload) as EvolutionReport;
      if (!parsed.run_id) {
        setErrorMessage("Report must include run_id.");
        return;
      }
      const next = [parsed, ...reports];
      persistReports(next);
      setImportPayload("");
      setStatusMessage("Report imported.");
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Invalid JSON.");
    }
  };

  const analytics = useMemo(() => {
    const byMode: Record<string, number[]> = {};
    reports.forEach((report) => {
      const mode = report.evolution_mode ?? "unknown";
      const score = Number(report.metrics?.best_score ?? report.summary_statistics?.best_score ?? 0);
      if (!byMode[mode]) byMode[mode] = [];
      byMode[mode].push(score);
    });
    const averages = Object.entries(byMode).map(([mode, scores]) => {
      const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
      return { mode, avg: Number(avg.toFixed(3)) };
    });
    return { averages };
  }, [reports]);

  const updateSettings = (updates: Record<string, unknown>) => {
    const next = { ...settings, ...updates };
    setSettings(next);
    saveJson("openevolve_reporting_settings", next);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Reporting Center</CardTitle>
          <CardDescription>Review and manage evolution reports.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <Tabs defaultValue="viewer">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="viewer">Report Viewer</TabsTrigger>
              <TabsTrigger value="analytics">Analytics Hub</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
            </TabsList>

            <TabsContent value="viewer" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Import Report</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Textarea
                    value={importPayload}
                    onChange={(event) => setImportPayload(event.target.value)}
                    rows={4}
                  />
                  <div className="flex gap-2">
                    <Button onClick={handleImport}>Import Report</Button>
                    <Button variant="outline" onClick={() => downloadJson("reports.json", reports)}>
                      Export All
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Reports</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {reports.length === 0 && (
                    <div className="text-sm text-muted-foreground">No reports available.</div>
                  )}
                  {reports.map((report) => (
                    <div key={report.run_id} className="rounded border p-3 space-y-1">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{report.run_id}</div>
                        <Button size="sm" variant="outline" onClick={() => setSelectedReport(report)}>
                          View
                        </Button>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Mode: {report.evolution_mode ?? "n/a"} · {report.timestamp ?? "n/a"}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {selectedReport ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Report Detail</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="secondary">{selectedReport.evolution_mode ?? "n/a"}</Badge>
                      <Badge variant="outline">{selectedReport.content_type ?? "n/a"}</Badge>
                    </div>
                    <pre className="rounded border p-2 text-xs whitespace-pre-wrap">
                      {JSON.stringify(selectedReport, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              ) : null}
            </TabsContent>

            <TabsContent value="analytics" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Mode Performance</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {analytics.averages.length === 0 && (
                    <div className="text-sm text-muted-foreground">No analytics available.</div>
                  )}
                  {analytics.averages.map((entry) => (
                    <div key={entry.mode} className="flex items-center justify-between rounded border p-2">
                      <span>{entry.mode}</span>
                      <Badge variant="secondary">{entry.avg}</Badge>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="settings" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Report Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>Auto-generate reports</Label>
                    <Switch
                      checked={settings.auto_generate_reports}
                      onCheckedChange={(value) => updateSettings({ auto_generate_reports: value })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>Detailed visualizations</Label>
                    <Switch
                      checked={settings.detailed_visualizations}
                      onCheckedChange={(value) => updateSettings({ detailed_visualizations: value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Preferred export formats (comma-separated)</Label>
                    <Input
                      value={settings.preferred_export_formats.join(", ")}
                      onChange={(event) =>
                        updateSettings({
                          preferred_export_formats: event.target.value
                            .split(",")
                            .map((item) => item.trim())
                            .filter(Boolean),
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Retention period</Label>
                    <Input
                      value={settings.report_retention}
                      onChange={(event) => updateSettings({ report_retention: event.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Default template</Label>
                    <Input
                      value={settings.default_report_template}
                      onChange={(event) => updateSettings({ default_report_template: event.target.value })}
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

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
exports.ReportingDashboardTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const switch_1 = require("@/components/ui/switch");
const readJson = (key, fallback) => {
    try {
        const raw = globalThis.localStorage?.getItem(key);
        if (!raw)
            return fallback;
        return JSON.parse(raw);
    }
    catch {
        return fallback;
    }
};
const saveJson = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, JSON.stringify(value));
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
const ReportingDashboardTab = () => {
    const [reports, setReports] = (0, react_1.useState)(() => readJson("openevolve_reports", []));
    const [selectedReport, setSelectedReport] = (0, react_1.useState)(null);
    const [importPayload, setImportPayload] = (0, react_1.useState)("");
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [settings, setSettings] = (0, react_1.useState)(() => readJson("openevolve_reporting_settings", {
        auto_generate_reports: true,
        detailed_visualizations: true,
        preferred_export_formats: ["PDF", "JSON"],
        report_retention: "3 months",
        default_report_template: "Standard Report",
    }));
    const persistReports = (next) => {
        setReports(next);
        saveJson("openevolve_reports", next);
    };
    const handleImport = () => {
        setErrorMessage(null);
        try {
            const parsed = JSON.parse(importPayload);
            if (!parsed.run_id) {
                setErrorMessage("Report must include run_id.");
                return;
            }
            const next = [parsed, ...reports];
            persistReports(next);
            setImportPayload("");
            setStatusMessage("Report imported.");
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Invalid JSON.");
        }
    };
    const analytics = (0, react_1.useMemo)(() => {
        const byMode = {};
        reports.forEach((report) => {
            const mode = report.evolution_mode ?? "unknown";
            const score = Number(report.metrics?.best_score ?? report.summary_statistics?.best_score ?? 0);
            if (!byMode[mode])
                byMode[mode] = [];
            byMode[mode].push(score);
        });
        const averages = Object.entries(byMode).map(([mode, scores]) => {
            const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
            return { mode, avg: Number(avg.toFixed(3)) };
        });
        return { averages };
    }, [reports]);
    const updateSettings = (updates) => {
        const next = { ...settings, ...updates };
        setSettings(next);
        saveJson("openevolve_reporting_settings", next);
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Reporting Center</card_1.CardTitle>
          <card_1.CardDescription>Review and manage evolution reports.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <tabs_1.Tabs defaultValue="viewer">
            <tabs_1.TabsList className="grid w-full grid-cols-3">
              <tabs_1.TabsTrigger value="viewer">Report Viewer</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="analytics">Analytics Hub</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="settings">Settings</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="viewer" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Import Report</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3">
                  <textarea_1.Textarea value={importPayload} onChange={(event) => setImportPayload(event.target.value)} rows={4}/>
                  <div className="flex gap-2">
                    <button_1.Button onClick={handleImport}>Import Report</button_1.Button>
                    <button_1.Button variant="outline" onClick={() => downloadJson("reports.json", reports)}>
                      Export All
                    </button_1.Button>
                  </div>
                </card_1.CardContent>
              </card_1.Card>

              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Reports</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2">
                  {reports.length === 0 && (<div className="text-sm text-muted-foreground">No reports available.</div>)}
                  {reports.map((report) => (<div key={report.run_id} className="rounded border p-3 space-y-1">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{report.run_id}</div>
                        <button_1.Button size="sm" variant="outline" onClick={() => setSelectedReport(report)}>
                          View
                        </button_1.Button>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Mode: {report.evolution_mode ?? "n/a"} · {report.timestamp ?? "n/a"}
                      </div>
                    </div>))}
                </card_1.CardContent>
              </card_1.Card>

              {selectedReport ? (<card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Report Detail</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2">
                    <div className="flex flex-wrap gap-2">
                      <badge_1.Badge variant="secondary">{selectedReport.evolution_mode ?? "n/a"}</badge_1.Badge>
                      <badge_1.Badge variant="outline">{selectedReport.content_type ?? "n/a"}</badge_1.Badge>
                    </div>
                    <pre className="rounded border p-2 text-xs whitespace-pre-wrap">
                      {JSON.stringify(selectedReport, null, 2)}
                    </pre>
                  </card_1.CardContent>
                </card_1.Card>) : null}
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="analytics" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Mode Performance</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2">
                  {analytics.averages.length === 0 && (<div className="text-sm text-muted-foreground">No analytics available.</div>)}
                  {analytics.averages.map((entry) => (<div key={entry.mode} className="flex items-center justify-between rounded border p-2">
                      <span>{entry.mode}</span>
                      <badge_1.Badge variant="secondary">{entry.avg}</badge_1.Badge>
                    </div>))}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="settings" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Report Settings</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label_1.Label>Auto-generate reports</label_1.Label>
                    <switch_1.Switch checked={settings.auto_generate_reports} onCheckedChange={(value) => updateSettings({ auto_generate_reports: value })}/>
                  </div>
                  <div className="flex items-center justify-between">
                    <label_1.Label>Detailed visualizations</label_1.Label>
                    <switch_1.Switch checked={settings.detailed_visualizations} onCheckedChange={(value) => updateSettings({ detailed_visualizations: value })}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Preferred export formats (comma-separated)</label_1.Label>
                    <input_1.Input value={settings.preferred_export_formats.join(", ")} onChange={(event) => updateSettings({
            preferred_export_formats: event.target.value
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean),
        })}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Retention period</label_1.Label>
                    <input_1.Input value={settings.report_retention} onChange={(event) => updateSettings({ report_retention: event.target.value })}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Default template</label_1.Label>
                    <input_1.Input value={settings.default_report_template} onChange={(event) => updateSettings({ default_report_template: event.target.value })}/>
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ReportingDashboardTab = ReportingDashboardTab;
//# sourceMappingURL=ReportingDashboardTab.js.map
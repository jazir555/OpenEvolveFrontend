import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "../../../lib/openevolveApi";
import type {
  MonitoringDashboardMetrics,
  MonitoringAlert,
  MonitoringService,
  MonitoringLogEntry,
} from "../../../lib/types";

const LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];

const formatValue = (value?: number | null, suffix = "") => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return `${value.toFixed(1)}${suffix}`;
};

export const SystemMonitoringTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [dashboard, setDashboard] = useState<MonitoringDashboardMetrics | null>(null);
  const [alerts, setAlerts] = useState<MonitoringAlert[]>([]);
  const [services, setServices] = useState<MonitoringService[]>([]);
  const [logs, setLogs] = useState<MonitoringLogEntry[]>([]);
  const [selectedService, setSelectedService] = useState("all");
  const [logLevels, setLogLevels] = useState<string[]>(["INFO", "WARNING", "ERROR"]);
  const [logSource, setLogSource] = useState<string>("");
  const [logLimit, setLogLimit] = useState(200);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [dashboardRes, alertsRes, servicesRes, logsRes] = await Promise.all([
        openevolveApi.getMonitoringDashboard(apiConfig),
        openevolveApi.getMonitoringAlerts(apiConfig),
        openevolveApi.getMonitoringServices(apiConfig),
        openevolveApi.getMonitoringLogs(logLimit, logSource || undefined, apiConfig),
      ]);
      setDashboard(dashboardRes);
      setAlerts(alertsRes.alerts || []);
      setServices(servicesRes.services || []);
      setLogs(logsRes.entries || []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load monitoring data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, [apiConfig.apiKey]);

  const filteredServices =
    selectedService === "all"
      ? services
      : services.filter((service) => service.name === selectedService);

  const filteredLogs = logs.filter((entry) =>
    logLevels.length
      ? logLevels.some((level) => entry.line.includes(level))
      : true,
  );

  const servicesUp = services.filter((service) => service.healthy).length;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>System Monitoring</CardTitle>
          <CardDescription>Service health checks, logs, and alerting signals.</CardDescription>
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
              Refresh Dashboard
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Services Up</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>
                  {servicesUp} / {services.length}
                </div>
                <div className="text-xs text-muted-foreground">healthy checks</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">CPU Usage</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>{formatValue(dashboard?.system?.system?.cpu_percent, "%")}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Memory Usage</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>{formatValue(dashboard?.system?.system?.memory_percent, "%")}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Active Alerts</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>{alerts.length}</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Service Status</CardTitle>
                <CardDescription>Health check output per subsystem.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex flex-col gap-2 md:flex-row md:items-center">
                  <div className="text-sm font-medium">Filter Service</div>
                  <Select value={selectedService} onValueChange={setSelectedService}>
                    <SelectTrigger className="w-[220px]">
                      <SelectValue placeholder="All services" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      {services.map((service) => (
                        <SelectItem key={service.name} value={service.name}>
                          {service.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                {filteredServices.length === 0 ? (
                  <div className="text-muted-foreground">No service checks available.</div>
                ) : (
                  filteredServices.map((service) => (
                    <div key={service.name} className="rounded border p-2">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{service.name}</div>
                        <Badge variant={service.healthy ? "default" : "secondary"}>
                          {service.status ?? "unknown"}
                        </Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Exec: {formatValue(service.execution_time, "s")} ·{" "}
                        {service.timestamp ?? "n/a"}
                      </div>
                      {service.error ? (
                        <div className="text-xs text-red-500">{service.error}</div>
                      ) : null}
                    </div>
                  ))
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Alerts</CardTitle>
                <CardDescription>Triggered alert rules from monitoring.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {alerts.length === 0 ? (
                  <div className="text-muted-foreground">No active alerts.</div>
                ) : (
                  alerts.map((alert, index) => (
                    <div key={`alert-${index}`} className="rounded border p-2">
                      <div className="font-semibold">{alert.name ?? "alert"}</div>
                      <div className="text-xs text-muted-foreground">
                        {alert.description ?? alert.metric_name ?? "threshold triggered"}
                      </div>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Logs</CardTitle>
              <CardDescription>Tail logs from known sources.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid gap-3 md:grid-cols-3">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Log Source</label>
                  <Input
                    value={logSource}
                    onChange={(event) => setLogSource(event.target.value)}
                    placeholder="backend_stdout.log"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Log Limit</label>
                  <Input
                    type="number"
                    value={logLimit}
                    onChange={(event) => setLogLimit(Number(event.target.value) || 200)}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Filter Levels</label>
                  <div className="flex flex-wrap gap-2">
                    {LOG_LEVELS.map((level) => (
                      <Button
                        key={level}
                        size="sm"
                        variant={logLevels.includes(level) ? "default" : "outline"}
                        onClick={() => {
                          setLogLevels((prev) =>
                            prev.includes(level)
                              ? prev.filter((value) => value !== level)
                              : [...prev, level],
                          );
                        }}
                      >
                        {level}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
              <Button variant="outline" onClick={refresh}>
                Reload Logs
              </Button>
              <Textarea
                className="min-h-[240px] text-xs"
                readOnly
                value={filteredLogs.map((entry) => `[${entry.source}] ${entry.line}`).join("\n")}
              />
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};

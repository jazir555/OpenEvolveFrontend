import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { openevolveApi } from "@/lib/openevolveApi";
import type { IcrOverview, IcrComponents, IcrRefinements } from "@/lib/types";

const parseJson = (value: string): Record<string, unknown> | undefined => {
  if (!value.trim()) return undefined;
  return JSON.parse(value);
};

export const IcrDashboardTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const [overview, setOverview] = useState<IcrOverview | null>(null);
  const [components, setComponents] = useState<IcrComponents | null>(null);
  const [refinements, setRefinements] = useState<IcrRefinements | null>(null);
  const [patterns, setPatterns] = useState<Record<string, unknown> | null>(null);
  const [vlm, setVlm] = useState<Record<string, unknown> | null>(null);
  const [heatmap, setHeatmap] = useState<{ points: Array<Record<string, unknown>>; total_snapshots: number } | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [vlmConfig, setVlmConfig] = useState<Record<string, unknown> | null>(null);

  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  const [eventReason, setEventReason] = useState("");
  const [calibOptionA, setCalibOptionA] = useState("");
  const [calibOptionB, setCalibOptionB] = useState("");
  const [calibRequestId, setCalibRequestId] = useState("");
  const [calibChoice, setCalibChoice] = useState("");

  const load = async <T,>(fn: () => Promise<T>, label: string): Promise<T | undefined> => {
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const response = await fn();
      setStatusMessage(`${label} loaded.`);
      return response;
    } catch (error: any) {
      setErrorMessage(error?.message ?? `${label} failed.`);
      return undefined;
    }
  };

  const runAction = async (
    fn: () => Promise<unknown>,
    label: string,
  ): Promise<void> => {
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const response = (await fn()) as Record<string, unknown>;
      setResult(response);
      setStatusMessage(`${label} completed.`);
    } catch (error: any) {
      setErrorMessage(error?.message ?? `${label} failed.`);
    }
  };

  return (
    <div className="space-y-6 bg-[#0d1117] text-gray-300 p-4 rounded">
      <div className="space-y-2">
        <Label className="text-gray-300">API Key</Label>
        <Input
          className="bg-[#0d1117] border-[#30363d] text-gray-300"
          value={apiKey}
          type="password"
          onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
              globalThis.localStorage?.setItem("openevolve_api_key", value);
            } catch {
              // ignore
            }
          }}
        />
      </div>

      {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
      {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrOverview(apiConfig), "Overview").then(setOverview)}
        >
          Overview
        </Button>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrComponents(apiConfig), "Components").then(setComponents)}
        >
          Components
        </Button>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrRefinements(apiConfig), "Refinements").then(setRefinements)}
        >
          Refinements
        </Button>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrAnalyticsPatterns(apiConfig), "Patterns").then(setPatterns)}
        >
          Patterns
        </Button>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrAnalyticsVlm(apiConfig), "VLM analytics").then(setVlm)}
        >
          VLM Analytics
        </Button>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrAnalyticsHeatmap(apiConfig), "Heatmap").then(setHeatmap)}
        >
          Heatmap
        </Button>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrConfig(apiConfig), "Config").then(setConfig)}
        >
          Config
        </Button>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrVlmConfig(apiConfig), "VLM config").then(setVlmConfig)}
        >
          VLM Config
        </Button>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() => load(() => openevolveApi.getIcrDashboard(apiConfig), "Dashboard").then(setResult)}
        >
          Dashboard
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {overview ? (
          <div className="rounded border border-[#30363d] p-3 text-sm space-y-1">
            <h4 className="font-semibold">Overview</h4>
            <div>Enabled: {String(overview.icr_enabled)}</div>
            <div>Total patterns: {overview.total_patterns}</div>
            <div>Success rate: {overview.overall_success_rate}</div>
            <div>Active components: {overview.active_components}</div>
            <div>Total refinements: {overview.total_refinements}</div>
          </div>
        ) : null}

        {heatmap ? (
          <div className="rounded border border-[#30363d] p-3 text-sm space-y-1">
            <h4 className="font-semibold">Heatmap</h4>
            <div>Points: {heatmap.points.length}</div>
            <div>Snapshots: {heatmap.total_snapshots}</div>
          </div>
        ) : null}
      </div>

      {components ? (
        <div className="rounded border border-[#30363d] p-4">
          <h4 className="mb-2 font-semibold">Components</h4>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(components, null, 2)}
          </pre>
        </div>
      ) : null}

      {patterns ? (
        <div className="rounded border border-[#30363d] p-4">
          <h4 className="mb-2 font-semibold">Pattern Analytics</h4>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(patterns, null, 2)}
          </pre>
        </div>
      ) : null}

      {vlm ? (
        <div className="rounded border border-[#30363d] p-4">
          <h4 className="mb-2 font-semibold">VLM Analytics</h4>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(vlm, null, 2)}
          </pre>
        </div>
      ) : null}

      {vlmConfig ? (
        <div className="rounded border border-[#30363d] p-4">
          <h4 className="mb-2 font-semibold">VLM Config</h4>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(vlmConfig, null, 2)}
          </pre>
        </div>
      ) : null}

      {config ? (
        <div className="rounded border border-[#30363d] p-4">
          <h4 className="mb-2 font-semibold">ICR Config</h4>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(config, null, 2)}
          </pre>
        </div>
      ) : null}

      {refinements ? (
        <div className="rounded border border-[#30363d] p-4">
          <h4 className="mb-2 font-semibold">Refinements</h4>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(refinements, null, 2)}
          </pre>
        </div>
      ) : null}

      <div className="rounded border border-[#30363d] p-4 space-y-3">
        <h4 className="font-semibold">Events &amp; Reward Calibration</h4>
        <div className="grid gap-3 md:grid-cols-2">
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Refinement reason"
            value={eventReason}
            onChange={(event) => setEventReason(event.target.value)}
          />
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(
                () =>
                  openevolveApi.emitIcrRefinementNeeded(
                    { reason: eventReason || null },
                    apiConfig,
                  ),
                "Emit refinement-needed",
              )
            }
          >
            Emit Refinement Event
          </Button>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Option A"
            value={calibOptionA}
            onChange={(event) => setCalibOptionA(event.target.value)}
          />
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Option B"
            value={calibOptionB}
            onChange={(event) => setCalibOptionB(event.target.value)}
          />
        </div>
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() =>
            runAction(
              () =>
                openevolveApi.requestIcrRewardCalibration(
                  { option_a: calibOptionA, option_b: calibOptionB },
                  apiConfig,
                ),
              "Request reward calibration",
            )
          }
        >
          Request Reward Calibration
        </Button>

        <div className="grid gap-3 md:grid-cols-2">
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Request ID"
            value={calibRequestId}
            onChange={(event) => setCalibRequestId(event.target.value)}
          />
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Choice"
            value={calibChoice}
            onChange={(event) => setCalibChoice(event.target.value)}
          />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() => runAction(() => openevolveApi.getIcrRewardCalibrationNext(apiConfig), "Next calibration")}
          >
            Next Calibration
          </Button>
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(
                () =>
                  openevolveApi.respondIcrRewardCalibration(
                    { request_id: calibRequestId || null, choice: calibChoice },
                    apiConfig,
                  ),
                "Respond calibration",
              )
            }
          >
            Respond Calibration
          </Button>
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(
                () => openevolveApi.getIcrRewardCalibrationResponse(calibRequestId, apiConfig),
                "Get calibration response",
              )
            }
          >
            Get Response
          </Button>
        </div>
      </div>

      {result ? (
        <div className="rounded border border-[#30363d] p-4">
          <h4 className="mb-2 font-semibold">Last Result</h4>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      ) : null}
    </div>
  );
};

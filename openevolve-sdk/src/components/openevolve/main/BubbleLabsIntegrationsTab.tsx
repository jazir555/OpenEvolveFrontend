import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { openevolveApi } from "@/lib/openevolveApi";
import type { BubbleLabsIntegrationsListResponse } from "@/lib/types";

interface IntegrationEntry {
  name: string;
  [key: string]: unknown;
}

const parseJson = (value: string): Record<string, unknown> | undefined => {
  if (!value.trim()) return undefined;
  return JSON.parse(value);
};

export const BubbleLabsIntegrationsTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [integrations, setIntegrations] = useState<IntegrationEntry[]>([]);
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [catalog, setCatalog] = useState<Record<string, unknown> | null>(null);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  const [forceDiscover, setForceDiscover] = useState(false);
  const [controlComponent, setControlComponent] = useState("");
  const [controlAction, setControlAction] = useState("");
  const [controlPayload, setControlPayload] = useState("{}");

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

  const refreshIntegrations = async () => {
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listBubblelabsIntegrations(apiConfig);
      const data = (response as BubbleLabsIntegrationsListResponse).integrations ?? [];
      setIntegrations(data as IntegrationEntry[]);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load integrations.");
    }
  };

  return (
    <div className="space-y-6 bg-[#0d1117] text-gray-300 p-4 rounded">
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
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
        <Button
          className="border-[#30363d]"
          variant="outline"
          onClick={refreshIntegrations}
        >
          Load Integrations
        </Button>
      </div>

      {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
      {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded border border-[#30363d] p-4">
          <h3 className="mb-3 font-semibold">Registered Integrations</h3>
          {integrations.length === 0 ? (
            <p className="text-sm text-gray-500">No integrations loaded.</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {integrations.map((integration) => (
                <li
                  key={integration.name}
                  className="flex items-center justify-between rounded border border-[#30363d] p-2"
                >
                  <span>{integration.name}</span>
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-[#30363d]"
                    onClick={() =>
                      runAction(
                        async () =>
                          openevolveApi.getBubblelabsIntegrationHealth(
                            integration.name,
                            apiConfig,
                          ),
                        `Health check for ${integration.name}`,
                      ).then(() => setHealth(result))
                    }
                  >
                    Health
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="rounded border border-[#30363d] p-4">
          <h3 className="mb-3 font-semibold">Integration Health</h3>
          {health ? (
            <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
              {JSON.stringify(health, null, 2)}
            </pre>
          ) : (
            <p className="text-sm text-gray-500">Run a health check.</p>
          )}
        </div>
      </div>

      <div className="rounded border border-[#30363d] p-4 space-y-4">
        <h3 className="font-semibold">Control Plane</h3>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() => runAction(() => openevolveApi.bubblelabsControlCatalog(apiConfig), "Catalog")}
          >
            Load Catalog
          </Button>
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(
                () => openevolveApi.bubblelabsControlDiscover({ force: forceDiscover }, apiConfig),
                "Discover",
              )
            }
          >
            Discover
          </Button>
        </div>

        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={forceDiscover}
            onChange={(event) => setForceDiscover(event.target.checked)}
          />
          Force discovery
        </label>

        {catalog ? (
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(catalog, null, 2)}
          </pre>
        ) : null}

        <div className="grid gap-3 md:grid-cols-2">
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Component"
            value={controlComponent}
            onChange={(event) => setControlComponent(event.target.value)}
          />
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Action"
            value={controlAction}
            onChange={(event) => setControlAction(event.target.value)}
          />
        </div>
        <Textarea
          className="bg-[#0d1117] border-[#30363d] text-gray-300"
          rows={4}
          placeholder='Payload JSON, e.g. {"key": "value"}'
          value={controlPayload}
          onChange={(event) => setControlPayload(event.target.value)}
        />
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() =>
            runAction(
              () =>
                openevolveApi.bubblelabsControlExecute(
                  {
                    component: controlComponent,
                    action: controlAction,
                    payload: parseJson(controlPayload) ?? {},
                  },
                  apiConfig,
                ),
              "Execute control action",
            )
          }
        >
          Execute Control Action
        </Button>
      </div>

      {result ? (
        <div className="rounded border border-[#30363d] p-4">
          <h3 className="mb-2 font-semibold">Last Result</h3>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      ) : null}
    </div>
  );
};

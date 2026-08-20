import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { openevolveApi } from "@/lib/openevolveApi";

const parseJson = (value: string): Record<string, unknown> | undefined => {
  if (!value.trim()) return undefined;
  return JSON.parse(value);
};

export const ResearchApprovalTab: React.FC = () => {
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
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  const [approvalWorkflowId, setApprovalWorkflowId] = useState("");
  const [approvalStageId, setApprovalStageId] = useState("1");
  const [approvalPayload, setApprovalPayload] = useState("{}");

  const [truthWorkflowId, setTruthWorkflowId] = useState("");
  const [truthPayload, setTruthPayload] = useState("{}");

  const [instanceId, setInstanceId] = useState("");
  const [instanceParameters, setInstanceParameters] = useState("{}");

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

      <div className="rounded border border-[#30363d] p-4 space-y-3">
        <h3 className="font-semibold">Research Approval</h3>
        <p className="text-xs text-gray-500">
          Approve a Research-Quest stage and trigger autonomous execution.
        </p>
        <div className="grid gap-3 md:grid-cols-2">
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Workflow ID"
            value={approvalWorkflowId}
            onChange={(event) => setApprovalWorkflowId(event.target.value)}
          />
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Stage ID"
            value={approvalStageId}
            onChange={(event) => setApprovalStageId(event.target.value)}
          />
        </div>
        <Textarea
          className="bg-[#0d1117] border-[#30363d] text-gray-300"
          rows={3}
          placeholder="Optional approval payload JSON"
          value={approvalPayload}
          onChange={(event) => setApprovalPayload(event.target.value)}
        />
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() =>
            runAction(
              () =>
                openevolveApi.submitResearchApproval(
                  approvalWorkflowId,
                  Number(approvalStageId),
                  parseJson(approvalPayload),
                  apiConfig,
                ),
              "Research approval",
            )
          }
        >
          Submit Research Approval
        </Button>
      </div>

      <div className="rounded border border-[#30363d] p-4 space-y-3">
        <h3 className="font-semibold">Truth Package</h3>
        <p className="text-xs text-gray-500">
          Generate a Truth Package binary trust artifact for a completed workflow.
        </p>
        <Input
          className="bg-[#0d1117] border-[#30363d] text-gray-300"
          placeholder="Workflow ID"
          value={truthWorkflowId}
          onChange={(event) => setTruthWorkflowId(event.target.value)}
        />
        <Textarea
          className="bg-[#0d1117] border-[#30363d] text-gray-300"
          rows={3}
          placeholder="Optional truth-package payload JSON"
          value={truthPayload}
          onChange={(event) => setTruthPayload(event.target.value)}
        />
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() =>
            runAction(
              () =>
                openevolveApi.createTruthPackage(
                  truthWorkflowId,
                  parseJson(truthPayload),
                  apiConfig,
                ),
              "Truth package",
            )
          }
        >
          Generate Truth Package
        </Button>
      </div>

      <div className="rounded border border-[#30363d] p-4 space-y-3">
        <h3 className="font-semibold">Instance Parameters</h3>
        <p className="text-xs text-gray-500">
          Sync parameters to a BubbleLabs workflow instance.
        </p>
        <Input
          className="bg-[#0d1117] border-[#30363d] text-gray-300"
          placeholder="Instance ID"
          value={instanceId}
          onChange={(event) => setInstanceId(event.target.value)}
        />
        <Textarea
          className="bg-[#0d1117] border-[#30363d] text-gray-300"
          rows={3}
          placeholder='Parameters JSON, e.g. {"temperature": 0.7}'
          value={instanceParameters}
          onChange={(event) => setInstanceParameters(event.target.value)}
        />
        <Button
          variant="outline"
          className="border-[#30363d]"
          onClick={() =>
            runAction(
              () =>
                openevolveApi.updateWorkflowInstanceParameters(
                  instanceId,
                  { parameters: parseJson(instanceParameters) ?? {} },
                  apiConfig,
                ),
              "Instance parameters",
            )
          }
        >
          Update Instance Parameters
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

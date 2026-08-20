import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { openevolveApi } from "@/lib/openevolveApi";

const parseJsonArray = (value: string): string[] => {
  if (!value.trim()) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
  }
};

export const Web3Tab: React.FC = () => {
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
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);

  const [projectPath, setProjectPath] = useState(".");
  const [runFuzzing, setRunFuzzing] = useState(true);
  const [slitherTimeout, setSlitherTimeout] = useState("240");
  const [forgeTimeout, setForgeTimeout] = useState("420");
  const [matchContract, setMatchContract] = useState("");
  const [matchTest, setMatchTest] = useState("");
  const [forkUrl, setForkUrl] = useState("");
  const [extraArgs, setExtraArgs] = useState("[]");

  const [statement, setStatement] = useState("");
  const [maxWithdrawExpr, setMaxWithdrawExpr] = useState("");
  const [verifyTranslation, setVerifyTranslation] = useState(true);
  const [nonNegative, setNonNegative] = useState(true);
  const [constraints, setConstraints] = useState("[]");
  const [witnessTimeout, setWitnessTimeout] = useState("10");

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
        <div className="flex gap-2">
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(async () => {
                const response = await openevolveApi.web3Status(apiConfig);
                setStatus(response);
                return response;
              }, "Status")
            }
          >
            Load Status
          </Button>
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() => runAction(() => openevolveApi.web3McpToolInventory(apiConfig), "MCP inventory")}
          >
            MCP Inventory
          </Button>
        </div>
      </div>

      {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
      {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

      {status ? (
        <div className="rounded border border-[#30363d] p-4">
          <h3 className="mb-2 font-semibold">Web3 Stack Status</h3>
          <pre className="rounded border border-[#30363d] p-2 text-xs whitespace-pre-wrap">
            {JSON.stringify(status, null, 2)}
          </pre>
        </div>
      ) : null}

      <div className="grid gap-3 md:grid-cols-2">
        <Input
          className="bg-[#0d1117] border-[#30363d] text-gray-300"
          placeholder="Project path"
          value={projectPath}
          onChange={(event) => setProjectPath(event.target.value)}
        />
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={runFuzzing}
            onChange={(event) => setRunFuzzing(event.target.checked)}
          />
          Run fuzzing
        </label>
      </div>

      <Tabs defaultValue="ingest" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="ingest">Ingest</TabsTrigger>
          <TabsTrigger value="invariants">Invariants</TabsTrigger>
          <TabsTrigger value="witness">Witness</TabsTrigger>
          <TabsTrigger value="audit">Audit</TabsTrigger>
        </TabsList>

        <TabsContent value="ingest" className="mt-4 space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <Input
              className="bg-[#0d1117] border-[#30363d] text-gray-300"
              placeholder="Slither timeout (s)"
              value={slitherTimeout}
              onChange={(event) => setSlitherTimeout(event.target.value)}
            />
            <Input
              className="bg-[#0d1117] border-[#30363d] text-gray-300"
              placeholder="Forge timeout (s)"
              value={forgeTimeout}
              onChange={(event) => setForgeTimeout(event.target.value)}
            />
          </div>
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(
                () =>
                  openevolveApi.web3Ingest(
                    {
                      project_path: projectPath,
                      run_fuzzing: runFuzzing,
                      slither_timeout_seconds: Number(slitherTimeout),
                      forge_timeout_seconds: Number(forgeTimeout),
                    },
                    apiConfig,
                  ),
                "Ingest stack",
              )
            }
          >
            Ingest Contract Stack
          </Button>

          <div className="grid gap-3 md:grid-cols-3">
            <Input
              className="bg-[#0d1117] border-[#30363d] text-gray-300"
              placeholder="Match contract"
              value={matchContract}
              onChange={(event) => setMatchContract(event.target.value)}
            />
            <Input
              className="bg-[#0d1117] border-[#30363d] text-gray-300"
              placeholder="Match test"
              value={matchTest}
              onChange={(event) => setMatchTest(event.target.value)}
            />
            <Input
              className="bg-[#0d1117] border-[#30363d] text-gray-300"
              placeholder="Fork URL"
              value={forkUrl}
              onChange={(event) => setForkUrl(event.target.value)}
            />
          </div>
          <Textarea
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            rows={2}
            placeholder='Extra args JSON, e.g. ["--via-ir"]'
            value={extraArgs}
            onChange={(event) => setExtraArgs(event.target.value)}
          />
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              className="border-[#30363d]"
              onClick={() =>
                runAction(
                  () =>
                    openevolveApi.web3IngestSlither(
                      {
                        project_path: projectPath,
                        timeout_seconds: Number(slitherTimeout),
                        extra_args: parseJsonArray(extraArgs),
                      },
                      apiConfig,
                    ),
                  "Slither",
                )
              }
            >
              Run Slither
            </Button>
            <Button
              variant="outline"
              className="border-[#30363d]"
              onClick={() =>
                runAction(
                  () =>
                    openevolveApi.web3IngestFoundry(
                      {
                        project_path: projectPath,
                        timeout_seconds: Number(forgeTimeout),
                        match_contract: matchContract || null,
                        match_test: matchTest || null,
                        fork_url: forkUrl || null,
                        extra_args: parseJsonArray(extraArgs),
                      },
                      apiConfig,
                    ),
                  "Foundry",
                )
              }
            >
              Run Foundry
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="invariants" className="mt-4 space-y-3">
          <Textarea
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            rows={3}
            placeholder="Solidity assignment statement"
            value={statement}
            onChange={(event) => setStatement(event.target.value)}
          />
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Max withdraw expr (optional)"
            value={maxWithdrawExpr}
            onChange={(event) => setMaxWithdrawExpr(event.target.value)}
          />
          <label className="flex items-center gap-4 text-sm">
            <span className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={nonNegative}
                onChange={(event) => setNonNegative(event.target.checked)}
              />
              Non-negative amount
            </span>
            <span className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={verifyTranslation}
                onChange={(event) => setVerifyTranslation(event.target.checked)}
              />
              Verify translation
            </span>
          </label>
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(
                () =>
                  openevolveApi.web3InvariantsTranslate(
                    {
                      statement,
                      non_negative_target: nonNegative,
                      max_withdraw_expr: maxWithdrawExpr || null,
                      verify_translation: verifyTranslation,
                      assume_non_negative_amount: nonNegative,
                    },
                    apiConfig,
                  ),
                "Translate invariant",
              )
            }
          >
            Translate Invariant
          </Button>
        </TabsContent>

        <TabsContent value="witness" className="mt-4 space-y-3">
          <Textarea
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            rows={3}
            placeholder='Additional constraints JSON array, e.g. ["x > 0"]'
            value={constraints}
            onChange={(event) => setConstraints(event.target.value)}
          />
          <Input
            className="bg-[#0d1117] border-[#30363d] text-gray-300"
            placeholder="Timeout (s)"
            value={witnessTimeout}
            onChange={(event) => setWitnessTimeout(event.target.value)}
          />
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(
                () =>
                  openevolveApi.web3ExploitsSymbolicWitness(
                    {
                      additional_constraints: parseJsonArray(constraints),
                      timeout_seconds: Number(witnessTimeout),
                    },
                    apiConfig,
                  ),
                "Symbolic witness",
              )
            }
          >
            Solve Symbolic Witness
          </Button>
        </TabsContent>

        <TabsContent value="audit" className="mt-4 space-y-3">
          <Button
            variant="outline"
            className="border-[#30363d]"
            onClick={() =>
              runAction(
                () =>
                  openevolveApi.web3AuditExploitVerification(
                    {
                      project_path: projectPath,
                      statement: statement || null,
                      run_fuzzing: runFuzzing,
                      verify_translation: verifyTranslation,
                      timeout_seconds: Number(witnessTimeout),
                      additional_constraints: parseJsonArray(constraints),
                      non_negative_target: nonNegative,
                      max_withdraw_expr: maxWithdrawExpr || null,
                      assume_non_negative_amount: nonNegative,
                    },
                    apiConfig,
                  ),
                "Audit exploit verification",
              )
            }
          >
            Run Audit / Exploit Verification
          </Button>
        </TabsContent>
      </Tabs>

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
